

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from decord import VideoReader, cpu
from tqdm import tqdm


def get_video_paths(root_path: Path, extensions: List[str]) -> List[str]:

    video_paths = []
    for extension in extensions:
        video_paths.extend(root_path.rglob(f"*{extension}"))
    return [str(path.resolve()) for path in video_paths]


def construct_derived_path(video_path: str, source_dir: str, target_dir: str, 
                          new_extension: str) -> str:

    path_obj = Path(video_path)
    # Replace the source_dir part of the path with target_dir
    parts = list(path_obj.parts)
    try:
        source_index = parts.index(source_dir)
        parts[source_index] = target_dir
    except ValueError:
        # If source_dir not found, insert target_dir before the last part
        parts.insert(-1, target_dir)
    
    new_path = Path(*parts).with_suffix(new_extension)
    return str(new_path)


def validate_file_exists(file_path: str, file_description: str) -> bool:

    if not os.path.exists(file_path):
        print(f"{file_description} not found: {file_path}")
        return False
    return True


def extract_video_metadata(video_path: str) -> Optional[Dict]:

    # Construct paths for associated files
    file_mappings = [
        ('videos', 'face_mask', '.png', 'mask_path'),
        ('videos', 'sep_pose_mask', '.png', 'sep_mask_border'),
        ('videos', 'sep_face_mask', '.png', 'sep_mask_face'), 
        ('videos', 'sep_lip_mask', '.png', 'sep_mask_lip'),
        ('videos', 'face_emb', '.pt', 'face_emb_path'),
        ('videos', 'audios', '.wav', 'audio_path'),
        ('videos', 'audio_emb', '.pt', 'vocals_emb_base_all')
    ]
    
    path_data = {}
    all_files_exist = True
    
    for source_dir, target_dir, ext, key in file_mappings:
        constructed_path = construct_derived_path(video_path, source_dir, target_dir, ext)
        path_data[key] = constructed_path
        
        if not validate_file_exists(constructed_path, key.replace('_', ' ').title()):
            all_files_exist = False
    
    if not all_files_exist:
        return None
    
    try:
        # Validate video frame count matches audio embedding length
        video_reader = VideoReader(video_path, ctx=cpu(0))
        audio_embeddings = torch.load(path_data['vocals_emb_base_all'])
        
        if abs(len(video_reader) - audio_embeddings.shape[0]) > 3:

            return None
        
        # Validate face embedding is not None
        face_embeddings = torch.load(path_data['face_emb_path'])
        if face_embeddings is None:

            return None
            
    except Exception as e:

        return None
    finally:
        # Clean up resources
        if 'video_reader' in locals():
            del video_reader
        if 'audio_embeddings' in locals():
            del audio_embeddings
        if 'face_embeddings' in locals():
            del face_embeddings
    
    # Combine all metadata
    metadata = {'video_path': str(video_path)}
    metadata.update(path_data)
    return metadata


def main():

    parser = argparse.ArgumentParser(
        description="Extract metadata from video files for training pipeline"
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
    
    # Set default meta_info_name if not provided
    if args.meta_info_name is None:
        args.meta_info_name = args.dataset_name
    
    # Find all video files
    video_directory = Path(args.root_path)
    video_files = get_video_paths(video_directory, ['.mp4'])
    
    if not video_files:
        print(f"No video files found in {video_directory}")
        return
    
    # Extract metadata with progress bar
    metadata_list = []
    for video_path in tqdm(video_files, desc="Processing videos"):
        metadata = extract_video_metadata(video_path)
        if metadata is not None:
            metadata_list.append(metadata)

    
    # Save metadata to JSON file
    output_path = Path(f"./data/{args.meta_info_name}_stage2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open('w', encoding='utf-8') as json_file:
        json.dump(metadata_list, json_file, indent=4, ensure_ascii=False)
    



if __name__ == "__main__":
    main()