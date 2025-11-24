

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoMetaInfoExtractor:
    """视频元信息提取器"""
    
    def __init__(self, root_path: Path, dataset_name: str, meta_info_name: Optional[str] = None):

        self.root_path = Path(root_path)
        self.dataset_name = dataset_name
        self.meta_info_name = meta_info_name or dataset_name
        self.output_dir = Path("./data")
        
    def validate_directory_structure(self) -> bool:

        image_dir = self.root_path / "images"
        if not image_dir.exists():
            logger.error(f"图像目录不存在: {image_dir}")
            return False
        return True
    
    def collect_video_directories(self) -> List[Path]:

        image_dir = self.root_path / "images"
        video_dirs = []
        
        for directory in image_dir.iterdir():
            if directory.is_dir():
                video_dirs.append(directory.resolve())
                logger.debug(f"找到视频目录: {directory.name}")
        
        logger.info(f"共找到 {len(video_dirs)} 个视频目录")
        return video_dirs
    
    def construct_meta_info_entry(self, video_dir: Path) -> Optional[Dict[str, str]]:

        # 构建掩码路径和面部嵌入路径
        base_path = str(video_dir).replace("images", "")
        mask_path = self.root_path / "face_mask" / f"{video_dir.name}.png"
        face_emb_path = self.root_path / "face_emb" / f"{video_dir.name}.pt"
        
        # 验证必要文件是否存在
        if not mask_path.exists():
            logger.warning(f"掩码文件不存在: {mask_path}")
            return None
        
        if not face_emb_path.exists():
            logger.warning(f"面部嵌入文件不存在: {face_emb_path}")
            return None
        
        # 验证面部嵌入文件是否有效
        try:
            face_embedding = torch.load(face_emb_path)
            if face_embedding is None:
                logger.warning(f"面部嵌入文件为空: {face_emb_path}")
                return None
        except Exception as e:
            logger.error(f"加载面部嵌入文件失败 {face_emb_path}: {e}")
            return None
        
        # 构建元信息条目
        meta_info = {
            "image_path": str(video_dir),
            "mask_path": str(mask_path),
            "face_emb": str(face_emb_path),
        }
        
        logger.debug(f"为视频目录 {video_dir.name} 创建元信息条目")
        return meta_info
    
    def extract_meta_information(self) -> List[Dict[str, str]]:

        if not self.validate_directory_structure():
            return []
        
        video_directories = self.collect_video_directories()
        meta_info_list = []
        
        logger.info("开始提取元信息...")
        for video_dir in video_directories:
            meta_info = self.construct_meta_info_entry(video_dir)
            if meta_info is not None:
                meta_info_list.append(meta_info)
        
        logger.info(f"成功提取 {len(meta_info_list)} 个有效元信息条目")
        return meta_info_list
    
    def save_meta_information(self, meta_info_list: List[Dict[str, str]]) -> Path:

        self.output_dir.mkdir(exist_ok=True)
        output_file = self.output_dir / f"{self.meta_info_name}_stage1.json"
        
        try:
            with output_file.open("w", encoding="utf-8") as file:
                json.dump(meta_info_list, file, indent=4, ensure_ascii=False)
            
            logger.info(f"元信息已保存至: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"保存元信息文件失败: {e}")
            raise
    
    def run_extraction(self) -> int:

        logger.info(f"开始为数据集 '{self.dataset_name}' 提取元信息")
        
        meta_info_list = self.extract_meta_information()
        
        if not meta_info_list:
            logger.warning("未找到任何有效的元信息条目")
            return 0
        
        self.save_meta_information(meta_info_list)
        logger.info(f"最终数据计数: {len(meta_info_list)}")
        
        return len(meta_info_list)


def main():
    """主函数：解析参数并执行元信息提取"""
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument(
        "-r", "--root_path",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "-n", "--dataset_name", 
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--meta_info_name",
        type=str
    )
    
    args = parser.parse_args()
    
    try:
        # 创建提取器并执行提取
        extractor = VideoMetaInfoExtractor(
            root_path=args.root_path,
            dataset_name=args.dataset_name,
            meta_info_name=args.meta_info_name
        )
        
        count = extractor.run_extraction()
        
        if count > 0:
            logger.info("元信息提取完成")
        else:
            logger.warning("元信息提取完成，但未找到有效数据")
            
    except Exception as e:
        logger.error(f"元信息提取过程失败: {e}")
        raise


if __name__ == "__main__":
    main()