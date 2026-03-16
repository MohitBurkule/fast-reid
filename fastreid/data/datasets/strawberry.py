# encoding: utf-8
"""
Strawberry ReID Dataset for fast-reid
Dataset structure:
    reid_crops/
        {farm_uuid}/
            {date}/
                {scan_type}/
                    {section_uuid}/
                        {track_id}/
                            frame_*.jpg
                            ...

Open-set ReID: Train and Val use DISJOINT track IDs.
The model must learn to match identities it has NEVER seen during training.
"""

import glob
import os
import random

from fastreid.data.datasets import DATASET_REGISTRY
from .bases import ImageDataset


@DATASET_REGISTRY.register()
class Strawberry(ImageDataset):
    """
    Strawberry ReID Dataset from crop directory structure

    Supports two formats:
    1. Original: crops/{track_id}/frame_*.jpg
    2. New: reid_crops/{farm_uuid}/{date}/{scan_type}/{section_uuid}/{track_id}/frame_*.jpg

    Open-set evaluation: Train and Val use completely different tracks.
    The model must learn to match identities it has NEVER seen during training.
    """
    dataset_dir = ""

    def __init__(self, root='datasets', **kwargs):
        if os.path.isdir(os.path.join(root, "crops")):
            self.data_dir = os.path.join(root, "crops")
            self.format = "original"
        elif os.path.isdir(os.path.join(root, "reid_crops")):
            self.data_dir = os.path.join(root, "reid_crops")
            self.format = "new"
        elif os.path.exists(root) and os.path.isdir(root):
            # Auto-detect format
            has_numeric_subdirs = any(d.isdigit() for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
            has_farm_uuids = any(len(d) == 36 for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

            if has_farm_uuids:
                self.data_dir = root
                self.format = "new"
            elif has_numeric_subdirs:
                self.data_dir = root
                self.format = "original"
            else:
                self.data_dir = os.path.join(root, "crops")
                self.format = "original"
        else:
            self.data_dir = root
            self.format = "original"

        train, query, gallery = self.load_data(**kwargs)

        super().__init__(train=train, query=query, gallery=gallery, **kwargs)

    def find_all_tracks(self):
        """
        Find all track directories based on format.

        Returns:
            List of (track_id, track_path) tuples
        """
        valid_tracks = []

        if self.format == "new":
            # New format: reid_crops/{farm_uuid}/{date}/{scan_type}/{section_uuid}/{track_id}/
            # Walk the directory tree and find all leaf directories with images
            for root_dir, dirs, files in os.walk(self.data_dir):
                # Check if this directory contains jpg files (it's a track directory)
                jpg_files = [f for f in files if f.endswith('.jpg')]
                if jpg_files:
                    # The track_id is the directory name
                    track_id_str = os.path.basename(root_dir)
                    try:
                        track_id = int(track_id_str)
                        valid_tracks.append((track_id, root_dir))
                    except ValueError:
                        # Skip non-numeric track directories
                        continue
        else:
            # Original format: crops/{track_id}/
            track_dirs = [d for d in os.listdir(self.data_dir)
                          if os.path.isdir(os.path.join(self.data_dir, d))]

            for track_id_str in track_dirs:
                if track_id_str.isdigit():
                    track_id = int(track_id_str)
                    track_path = os.path.join(self.data_dir, track_id_str)
                    valid_tracks.append((track_id, track_path))

        return valid_tracks

    def load_data(self, min_images=4, train_ratio=0.8, seed=42, **kwargs):
        """
        Load strawberry crop data with proper open-set split.

        Args:
            min_images: minimum images per track to include
            train_ratio: fraction of tracks for training (rest for val)
            seed: random seed for reproducible splits

        Returns:
            train_data: all images from train tracks
            query_data: 1 image per val track
            gallery_data: remaining images per val track
        """
        train_data = []
        query_data = []
        gallery_data = []

        if not os.path.exists(self.data_dir):
            raise ValueError(f"Crops directory not found: {self.data_dir}")

        # Find all track directories
        all_tracks = self.find_all_tracks()

        print(f"[Strawberry] Found {len(all_tracks)} total tracks in {self.format} format")

        # Filter tracks with enough images
        valid_tracks = []
        for track_id, track_path in all_tracks:
            images = sorted(glob.glob(os.path.join(track_path, "*.jpg")))
            if len(images) >= min_images:
                valid_tracks.append((track_id, images))

        print(f"[Strawberry] Found {len(valid_tracks)} valid tracks (min images={min_images})")

        # Sort by track_id for reproducibility
        valid_tracks = sorted(valid_tracks, key=lambda x: x[0])

        # Shuffle and split tracks into train/val
        random.seed(seed)
        random.shuffle(valid_tracks)

        split_idx = int(len(valid_tracks) * train_ratio)
        train_tracks = valid_tracks[:split_idx]
        val_tracks = valid_tracks[split_idx:]

        print(f"[Strawberry] Train tracks: {len(train_tracks)}, Val tracks: {len(val_tracks)}")
        print(f"[Strawberry] DISJOINT: Train and Val have NO overlapping identities!")

        # Train data: ALL images from train tracks
        for pid, (track_id, images) in enumerate(train_tracks):
            for img_path in images:
                train_data.append((img_path, pid, 0))  # camid=0

        # Val data: query/gallery split from val tracks (disjoint IDs from train)
        for pid, (track_id, images) in enumerate(val_tracks):
            val_pid = pid + len(train_tracks)  # offset IDs
            query_data.append((images[0], val_pid, 0))  # camid=0 for query
            for img in images[1:]:
                gallery_data.append((img, val_pid, 1))  # camid=1 for gallery

        print(f"[Strawberry] Train: {len(train_data)} images from {len(train_tracks)} identities (IDs 0-{len(train_tracks)-1})")
        print(f"[Strawberry] Query: {len(query_data)} images from {len(val_tracks)} identities (IDs {len(train_tracks)}-{len(train_tracks)+len(val_tracks)-1})")
        print(f"[Strawberry] Gallery: {len(gallery_data)} images")

        return train_data, query_data, gallery_data
