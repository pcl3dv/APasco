

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from torch import nn

from apasco.animate.face_animate import FaceAnimatePipeline
from apasco.datasets.audio_processor import AudioProcessor
from apasco.datasets.image_processor import ImageProcessor
from apasco.models.audio_proj import AudioProjModel
from apasco.models.face_locator import FaceLocator
from apasco.models.image_proj import ImageProjModel
from apacos.models.unet_2d_condition import UNet2DConditionModel
from apacos.models.unet_3d import UNet3DConditionModel
from apacos.utils.config import filter_non_none
from apacos.utils.util import tensor_to_video


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferenceNetwork(nn.Module):

    
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        image_proj: ImageProjModel,
        audio_proj: AudioProjModel,
    ):

        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.image_proj = image_proj
        self.audio_proj = audio_proj

    def forward(self):
        """空函数以覆盖nn.Module的抽象函数"""

    def get_modules(self) -> Dict[str, nn.Module]:

        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "image_proj": self.image_proj,
            "audio_proj": self.audio_proj,
        }


class InferenceProcessor:

    
    def __init__(self, config_path: str, cli_args: Dict):

        self.config = self._load_config(config_path, cli_args)
        self.device = self._setup_device()
        self.weight_dtype = self._get_weight_dtype()
        self.network = None
        self.pipeline = None
        
    def _load_config(self, config_path: str, cli_args: Dict) -> OmegaConf:

        config = OmegaConf.load(config_path)
        filtered_args = filter_non_none(cli_args)
        return OmegaConf.merge(config, filtered_args)
    
    def _setup_device(self) -> torch.device:

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_weight_dtype(self) -> torch.dtype:

        dtype_mapping = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32
        }
        return dtype_mapping.get(self.config.weight_dtype, torch.float32)
    
    def _create_scheduler(self) -> DDIMScheduler:
        """创建噪声调度器"""
        sched_kwargs = OmegaConf.to_container(self.config.noise_scheduler_kwargs)
        
        if self.config.get("enable_zero_snr", False):
            sched_kwargs.update({
                "rescale_betas_zero_snr": True,
                "timestep_spacing": "trailing",
                "prediction_type": "v_prediction",
            })
            
        return DDIMScheduler(**sched_kwargs)
    
    def _load_models(self) -> Tuple:

        # 创建调度器
        scheduler = self._create_scheduler()
        
        # 加载VAE
        vae = AutoencoderKL.from_pretrained(self.config.vae.model_path)
        
        # 加载参考UNet
        reference_unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path, subfolder="unet"
        )
        
        # 加载去噪UNet
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            self.config.base_model_path,
            self.config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                self.config.unet_additional_kwargs
            ),
            use_landmark=False,
        )
        
        # 加载其他模块
        face_locator = FaceLocator(conditioning_embedding_channels=320)
        
        image_proj = ImageProjModel(
            cross_attention_dim=denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        )
        
        audio_proj = AudioProjModel(
            seq_len=5,
            blocks=12,  # 使用wav2vec的12层隐藏状态
            channels=768,  # 音频嵌入通道数
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        ).to(device=self.device, dtype=self.weight_dtype)
        
        return vae, reference_unet, denoising_unet, face_locator, image_proj, audio_proj, scheduler
    
    def _freeze_models(self, *models) -> None:

        for model in models:
            if model is not None:
                model.requires_grad_(False)
    
    def _load_checkpoint(self, network: InferenceNetwork) -> None:

        checkpoint_path = Path(self.config.audio_ckpt_dir) / "net.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        missing_keys, unexpected_keys = network.load_state_dict(state_dict)
        
        if missing_keys:
            logger.warning(f"缺少的键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"意外的键: {unexpected_keys}")
            
        if not missing_keys and not unexpected_keys:
            logger.info(f"成功加载检查点: {checkpoint_path}")
        else:
            logger.warning(f"检查点加载完成，但存在键不匹配")
    
    def _process_audio_embedding(self, audio_emb: torch.Tensor) -> torch.Tensor:

        concatenated_tensors = []
        
        for i in range(audio_emb.shape[0]):
            # 获取当前帧及其前后各两帧的嵌入
            indices = [max(min(i + j, audio_emb.shape[0] - 1), 0) 
                      for j in range(-2, 3)]
            frame_embeddings = [audio_emb[idx] for idx in indices]
            concatenated_tensors.append(torch.stack(frame_embeddings, dim=0))
        
        return torch.stack(concatenated_tensors, dim=0)
    
    def _prepare_image_data(self) -> Tuple:

        img_size = (self.config.data.source_image.width, 
                   self.config.data.source_image.height)
        face_analysis_model_path = self.config.face_analysis.model_path
        
        with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
            return image_processor.preprocess(
                self.config.source_image,
                self.config.save_path,
                self.config.face_expand_ratio
            )
    
    def _prepare_audio_data(self, clip_length: int) -> Tuple[torch.Tensor, int]:

        sample_rate = self.config.data.driving_audio.sample_rate
        if sample_rate != 16000:
            raise ValueError("音频采样率必须为16000")
            
        fps = self.config.data.export_video.fps
        wav2vec_model_path = self.config.wav2vec.model_path
        wav2vec_only_last_features = self.config.wav2vec.features == "last"
        audio_separator_model_file = self.config.audio_separator.model_path
        
        with AudioProcessor(
            sample_rate,
            fps,
            wav2vec_model_path,
            wav2vec_only_last_features,
            os.path.dirname(audio_separator_model_file),
            os.path.basename(audio_separator_model_file),
            os.path.join(self.config.save_path, "audio_preprocess")
        ) as audio_processor:
            return audio_processor.preprocess(
                self.config.driving_audio, clip_length
            )
    
    def _prepare_tensors(self, source_data: Tuple, audio_emb: torch.Tensor, 
                        clip_length: int) -> Dict[str, torch.Tensor]:

        (source_image_pixels, source_image_face_region, source_image_face_emb,
         source_image_full_mask, source_image_face_mask, source_image_lip_mask) = source_data
        
        # 处理音频嵌入
        processed_audio_emb = self._process_audio_embedding(audio_emb)
        
        # 准备图像张量
        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)
        
        # 准备掩码张量
        mask_tensors = {}
        masks_data = [
            (source_image_full_mask, "full_mask"),
            (source_image_face_mask, "face_mask"), 
            (source_image_lip_mask, "lip_mask")
        ]
        
        for masks, mask_type in masks_data:
            mask_tensors[mask_type] = [
                mask.repeat(clip_length, 1) for mask in masks
            ]
        
        return {
            "audio_emb": processed_audio_emb,
            "image_pixels": source_image_pixels,
            "face_region": source_image_face_region,
            "face_emb": source_image_face_emb,
            **mask_tensors
        }
    
    def run_inference(self) -> str:

        
        # 确保保存目录存在
        save_path = Path(self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 准备运动尺度
        motion_scale = [
            self.config.pose_weight, 
            self.config.face_weight, 
            self.config.lip_weight
        ]
        
        # 准备图像数据
        source_data = self._prepare_image_data()
        
        # 准备音频数据
        clip_length = self.config.data.n_sample_frames
        audio_emb, audio_length = self._prepare_audio_data(clip_length)
        
        # 加载模型
        models = self._load_models()
        vae, reference_unet, denoising_unet, face_locator, image_proj, audio_proj, scheduler = models
        
        # 冻结模型
        self._freeze_models(vae, image_proj, reference_unet, denoising_unet, face_locator, audio_proj)
        
        # 启用梯度检查点
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()
        
        # 创建网络并加载检查点
        self.network = InferenceNetwork(
            reference_unet, denoising_unet, face_locator, image_proj, audio_proj
        )
        self._load_checkpoint(self.network)
        
        # 创建推理管道
        self.pipeline = FaceAnimatePipeline(
            vae=vae,
            reference_unet=self.network.reference_unet,
            denoising_unet=self.network.denoising_unet,
            face_locator=self.network.face_locator,
            scheduler=scheduler,
            image_proj=self.network.image_proj,
        )
        self.pipeline.to(device=self.device, dtype=self.weight_dtype)
        
        # 准备张量数据
        tensors = self._prepare_tensors(source_data, audio_emb, clip_length)
        
        # 运行推理循环
        result_tensor = self._run_inference_loop(tensors, clip_length, audio_length, motion_scale)
        
        # 保存结果
        output_file = self.config.output
        tensor_to_video(result_tensor, output_file, self.config.driving_audio)
        
        return output_file
    
    def _run_inference_loop(self, tensors: Dict[str, torch.Tensor], clip_length: int,
                          audio_length: int, motion_scale: List[float]) -> torch.Tensor:
        audio_emb = tensors["audio_emb"]
        times = audio_emb.shape[0] // clip_length
        
        tensor_results = []
        generator = torch.manual_seed(42)
        img_size = (self.config.data.source_image.width, 
                   self.config.data.source_image.height)
        
        for t in range(times):
            logger.info(f"处理片段 [{t+1}/{times}]")
            
            # 准备参考图像
            pixel_values_ref_img = self._prepare_reference_image(
                tensors["image_pixels"], tensor_results, t, clip_length
            )
            
            # 处理音频张量
            audio_tensor = self._process_audio_tensor(audio_emb, t, clip_length)
            
            # 运行管道推理
            pipeline_output = self.pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=tensors["face_emb"],
                face_mask=tensors["face_region"],
                pixel_values_full_mask=tensors["full_mask"],
                pixel_values_face_mask=tensors["face_mask"],
                pixel_values_lip_mask=tensors["lip_mask"],
                width=img_size[0],
                height=img_size[1],
                video_length=clip_length,
                num_inference_steps=self.config.inference_steps,
                guidance_scale=self.config.cfg_scale,
                generator=generator,
                motion_scale=motion_scale,
            )
            
            tensor_results.append(pipeline_output.videos)
        
        # 合并所有结果
        final_tensor = torch.cat(tensor_results, dim=2)
        final_tensor = final_tensor.squeeze(0)
        return final_tensor[:, :audio_length]
    
    def _prepare_reference_image(self, source_image: torch.Tensor, 
                               previous_results: List, current_iter: int,
                               clip_length: int) -> torch.Tensor:
        """准备参考图像"""
        n_motion_frames = self.config.data.n_motion_frames
        
        if current_iter == 0:
            # 第一次迭代使用零运动帧
            motion_zeros = source_image.repeat(n_motion_frames, 1, 1, 1)
            motion_zeros = motion_zeros.to(
                dtype=source_image.dtype, device=source_image.device
            )
            return torch.cat([source_image, motion_zeros], dim=0).unsqueeze(0)
        else:
            # 使用前一次迭代的结果作为运动帧
            motion_frames = previous_results[-1][0]
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[-n_motion_frames:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(
                dtype=source_image.dtype, device=source_image.device
            )
            return torch.cat([source_image, motion_frames], dim=0).unsqueeze(0)
    
    def _process_audio_tensor(self, audio_emb: torch.Tensor, iteration: int,
                            clip_length: int) -> torch.Tensor:
        """处理音频张量"""
        start_idx = iteration * clip_length
        end_idx = min((iteration + 1) * clip_length, audio_emb.shape[0])
        
        audio_tensor = audio_emb[start_idx:end_idx].unsqueeze(0)
        audio_tensor = audio_tensor.to(
            device=self.network.audio_proj.device,
            dtype=self.network.audio_proj.dtype
        )
        return self.network.audio_proj(audio_tensor)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument(
        "-c", "--config", 
        default="configs/inference/default.yaml"
    )
    parser.add_argument(
        "--source_image", 
        type=str
    )
    parser.add_argument(
        "--driving_audio", 
        type=str
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=".cache/output.mp4"
    )
    parser.add_argument(
        "--pose_weight", 
        type=float
    )
    parser.add_argument(
        "--face_weight", 
        type=float
    )
    parser.add_argument(
        "--lip_weight", 
        type=float
    )
    parser.add_argument(
        "--face_expand_ratio", 
        type=float
    )
    parser.add_argument(
        "--audio_ckpt_dir", "--checkpoint", 
        type=str
    )
    
    args = parser.parse_args()
    
    try:
        processor = InferenceProcessor(args.config, vars(args))
        output_path = processor.run_inference()
    except Exception as e:
        logger.error(f"推理过程失败: {e}")
        raise


if __name__ == "__main__":
    main()