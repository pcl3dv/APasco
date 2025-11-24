
import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import torch
from tqdm import tqdm

from apasco.datasets.audio_processor import AudioProcessor
from apasco.datasets.image_processor import ImageProcessorForDataProcessing
from apasco.utils.util import convert_video_to_images, extract_audio_from_videos

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoDataProcessor:

    
    def __init__(self, output_dir: Path, step: int):

        self.output_dir = output_dir
        self.step = step
        self.image_processor = None
        self.audio_processor = None
        
        self._initialize_processors()
    
    def _initialize_processors(self) -> None:

        face_analysis_model_path = "pretrained_models/face_analysis"
        landmark_model_path = "pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task"
        
        # 初始化图像处理器
        self.image_processor = ImageProcessorForDataProcessing(
            face_analysis_model_path, landmark_model_path, self.step
        )
        
        # 仅在步骤2初始化音频处理器
        if self.step == 2:
            audio_separator_model_file = "pretrained_models/audio_separator/Kim_Vocal_2.onnx"
            wav2vec_model_path = 'pretrained_models/wav2vec/wav2vec2-base-960h'
            
            self.audio_processor = AudioProcessor(
                sample_rate=16000,
                hop_length=25,
                wav2vec_model_path=wav2vec_model_path,
                use_gpu=False,
                model_dir=os.path.dirname(audio_separator_model_file),
                model_file=os.path.basename(audio_separator_model_file),
                vocal_output_dir=os.path.join(self.output_dir, "vocals"),
            )
    
    def setup_processing_directories(self, video_path: Path) -> Dict[str, Path]:

        base_dir = video_path.parent.parent
        directory_structure = {
            "face_mask": base_dir / "face_mask",
            "sep_pose_mask": base_dir / "sep_pose_mask", 
            "sep_face_mask": base_dir / "sep_face_mask",
            "sep_lip_mask": base_dir / "sep_lip_mask",
            "face_emb": base_dir / "face_emb",
            "audio_emb": base_dir / "audio_emb"
        }
        
        # 创建所有需要的目录
        for directory in directory_structure.values():
            directory.mkdir(parents=True, exist_ok=True)
            
        return directory_structure
    
    def process_video(self, video_path: Path) -> None:

        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return
            
        logger.info(f"开始处理视频: {video_path.name}")
        directories = self.setup_processing_directories(video_path)
        
        try:
            if self.step == 1:
                self._process_step_1(video_path, directories)
            else:
                self._process_step_2(video_path, directories)
                
            logger.info(f"视频处理完成: {video_path.name}")
            
        except Exception as error:
            logger.error(f"处理视频 {video_path} 时发生错误: {error}")
            raise
    
    def _process_step_1(self, video_path: Path, directories: Dict[str, Path]) -> None:

        # 提取视频帧
        images_dir = self.output_dir / 'images' / video_path.stem
        images_dir.mkdir(parents=True, exist_ok=True)
        images_output_path = convert_video_to_images(video_path, images_dir)
        logger.info(f"视频帧已保存至: {images_output_path}")
        
        # 提取音频
        audio_dir = self.output_dir / 'audios'
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_output_path = audio_dir / f'{video_path.stem}.wav'
        extract_audio_from_videos(video_path, audio_output_path)
        logger.info(f"音频已提取至: {audio_output_path}")
        
        # 生成各种掩码
        face_mask, _, sep_pose_mask, sep_face_mask, sep_lip_mask = (
            self.image_processor.preprocess(images_output_path)
        )
        
        # 保存掩码文件
        self._save_mask_files(video_path, directories, {
            "face_mask": face_mask,
            "sep_pose_mask": sep_pose_mask,
            "sep_face_mask": sep_face_mask, 
            "sep_lip_mask": sep_lip_mask
        })
    
    def _process_step_2(self, video_path: Path, directories: Dict[str, Path]) -> None:

        images_dir = self.output_dir / "images" / video_path.stem
        audio_path = self.output_dir / "audios" / f"{video_path.stem}.wav"
        
        # 提取面部嵌入
        _, face_embedding, _, _, _ = self.image_processor.preprocess(images_dir)
        embedding_path = directories["face_emb"] / f"{video_path.stem}.pt"
        torch.save(face_embedding, str(embedding_path))
        
        # 提取音频嵌入
        audio_embedding, _ = self.audio_processor.preprocess(audio_path)
        audio_embedding_path = directories["audio_emb"] / f"{video_path.stem}.pt"
        torch.save(audio_embedding, str(audio_embedding_path))
    
    def _save_mask_files(self, video_path: Path, directories: Dict[str, Path], 
                        masks: Dict[str, torch.Tensor]) -> None:

        for mask_type, mask_data in masks.items():
            if mask_data is not None:
                output_path = directories[mask_type] / f"{video_path.stem}.png"
                cv2.imwrite(str(output_path), mask_data)


def get_video_file_paths(source_dir: Path, parallelism: int = 1, 
                        rank: int = 0) -> List[Path]:

    if not source_dir.exists():
        raise ValueError(f"源目录不存在: {source_dir}")
    
    video_files = [
        item for item in sorted(source_dir.iterdir())
        if item.is_file() and item.suffix.lower() == '.mp4'
    ]
    
    # 根据并行配置分配文件
    return [
        video_files[i] for i in range(len(video_files))
        if i % parallelism == rank
    ]


def process_video_collection(video_paths: List[Path], output_dir: Path, 
                           step: int) -> None:

    if not video_paths:
        logger.warning("没有找到要处理的视频文件")
        return
    
    processor = VideoDataProcessor(output_dir, step)
    
    progress_description = f"处理步骤 {step} 的视频"
    for video_path in tqdm(video_paths, desc=progress_description):
        processor.process_video(video_path)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument(
        "-i", "--input_dir", 
        type=Path,
        required=True
    )
    
    parser.add_argument(
        "-o", "--output_dir", 
        type=Path
    )
    
    parser.add_argument(
        "-s", "--step", 
        type=int,
        choices=[1, 2],
        default=1
    )
    
    parser.add_argument(
        "-p", "--parallelism", 
        type=int,
        default=1
    )
    
    parser.add_argument(
        "-r", "--rank", 
        type=int,
        default=0
    )
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent
    
    # 获取视频文件列表
    try:
        video_files = get_video_file_paths(
            args.input_dir, args.parallelism, args.rank
        )
    except ValueError as e:
        logger.error(e)
        return
    
    # 处理视频
    process_video_collection(video_files, args.output_dir, args.step)
    logger.info("所有视频处理完成")


if __name__ == "__main__":
    main()