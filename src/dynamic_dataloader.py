"""
Created by: Anshuman Dewangan
Date: 2021

Description: Loads data from raw image and XML files
"""

# Torch imports
import pytorch_lightning as pl
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# Other package imports
import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch

# File imports
import util_fns


#####################
## Data Module
#####################

class DynamicDataModule(pl.LightningDataModule):
    def __init__(self, 
                 omit_list=None,
                 mask_omit_images=False,
                 is_object_detection=False,
                 is_maskrcnn=False,
                 is_background_removal=False,
                 
                 raw_data_path=None, 
                 labels_path=None, 
                 tile_label_stats_path=None,
                 metadata_path='./data/metadata.pkl',
                 optical_flow_path=None,
                 
                 train_split_path=None,
                 val_split_path=None,
                 test_split_path=None,
                 load_images_from_split=False,
                 
                 train_split_size=0.7,
                 test_split_size=0.15,
                 batch_size=1, 
                 num_workers=0, 
                 series_length=1, 
                 add_base_flow=False,
                 time_range=(-2400, 2400), 
                 
                 original_dimensions = (1536, 2016),
                 resize_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224, 224),
                 tile_overlap = 0,
                 pre_tile = True,
                 smoke_threshold = 250,
                 num_tile_samples = 0,
                 
                 flip_augment = True,
                 resize_crop_augment = True,
                 blur_augment = True,
                 color_augment = True,
                 brightness_contrast_augment = True):
        """
        Args: (see arg descriptions in main.py)
        Attributes:
            - self.train_split (list): list of image names to be used for train dataloader
            - self.val_split (list): list of image names to be used for val dataloader
            - self.test_split (list): list of image names to be used for test dataloader
            - self.has_setup (bool): if setup has already occurred to prevent from doing twice
        """
        super().__init__()
        
        self.omit_list = omit_list
        self.mask_omit_images = mask_omit_images
        self.is_object_detection = is_object_detection
        self.is_maskrcnn = is_maskrcnn
        self.is_background_removal = is_background_removal
           
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.tile_label_stats_path = tile_label_stats_path
        self.tile_label_stats = None
        if tile_label_stats_path is not None:
            with open(tile_label_stats_path, 'rb') as stats_file:
                self.tile_label_stats = pickle.load(stats_file)
            if not isinstance(self.tile_label_stats, dict):
                raise TypeError('tile_label_stats_path must contain a dictionary keyed by image name.')
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        self.optical_flow_path = optical_flow_path
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        self.load_images_from_split = load_images_from_split
        
        self.train_split_size = train_split_size
        self.test_split_size = test_split_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.series_length = series_length
        self.add_base_flow = add_base_flow
        self.time_range = time_range
        
        self.original_dimensions = original_dimensions
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        
        self.tile_dimensions = tile_dimensions
        self.tile_overlap = tile_overlap
        self.pre_tile = pre_tile
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.resize_crop_augment = resize_crop_augment
        self.blur_augment = blur_augment
        self.color_augment = color_augment
        self.brightness_contrast_augment = brightness_contrast_augment

        if self.tile_label_stats is not None:
            if self.is_object_detection:
                raise ValueError('Precomputed tile labels are only supported by tile-classification models.')
            if self.num_tile_samples > 0:
                raise ValueError('Precomputed tile labels do not support --num-tile-samples.')
            paper_geometry = ((1392, 1856), 1040, (224, 224), 20)
            configured_geometry = (
                self.resize_dimensions,
                self.crop_height,
                self.tile_dimensions,
                self.tile_overlap,
            )
            if configured_geometry != paper_geometry:
                raise ValueError(
                    'labels_stats_90overlap.pkl requires paper geometry: '
                    'resize=(1392, 1856), crop_height=1040, '
                    'tile_dimensions=(224, 224), tile_overlap=20.'
                )
            if self.resize_crop_augment:
                raise ValueError(
                    'Precomputed tile labels require fixed paper geometry. '
                    'Disable random crop jitter with --no-resize-crop-augment.'
                )
        
        self.has_setup = False
        
    def setup(self, stage=None, log_dir=None):
        """
        Args:
            - log_dir (str): logging directory to save train/val/test splits 
        """
        if self.has_setup: return
        print("Setting Up Data...")

        # Determine which images to omit
        self.omit_images_list = []
        if self.omit_list is not None:
            for key in self.omit_list:
                self.omit_images_list.extend(self.metadata[key])
                        
        is_split_given = self.train_split_path is not None and self.val_split_path is not None and self.test_split_path is not None
        
        # If split is given, load existing splits
        if is_split_given:
            train_list = np.loadtxt(self.train_split_path, dtype=str)
            val_list = np.loadtxt(self.val_split_path, dtype=str)
            test_list = np.loadtxt(self.test_split_path, dtype=str)
                
        # If load_images_from_split, load images exactly from provided splits
        if is_split_given and self.load_images_from_split:
            self.train_split = [util_fns.get_image_name(item) for item in train_list]
            self.val_split   = [util_fns.get_image_name(item) for item in val_list]
            self.test_split  = [util_fns.get_image_name(item) for item in test_list]
            
            # Recreate fire_to_images and image_series
            self.metadata['fire_to_images'] = util_fns.generate_fire_to_images_from_splits([self.train_split, self.val_split, self.test_split])
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length, self.add_base_flow) 
        
        else:
            # If split is provided, extract fires from split
            if is_split_given:
                train_fires = {util_fns.get_fire_name(item) for item in train_list}
                val_fires   = {util_fns.get_fire_name(item) for item in val_list}
                test_fires  = {util_fns.get_fire_name(item) for item in test_list}
            # Else if split is not provided, randomly create our own splits
            else:
                train_fires, val_fires = train_test_split(self.metadata['eligible_fires'], test_size=(1-self.train_split_size))
                val_fires, test_fires = train_test_split(val_fires, test_size=self.test_split_size/(1-self.train_split_size))
                train_fires += self.metadata['train_only_fires']
                        
            # Save arrays representing series of images
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length, self.add_base_flow)
            
            # Shorten fire_to_images to relevant time frame
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(train_fires, self.metadata['fire_to_images'], self.time_range, self.series_length, self.add_base_flow, self.optical_flow_path)
            # Shorten val/test only by series_length
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(val_fires, self.metadata['fire_to_images'], (-2400,2400), self.series_length,  self.add_base_flow, self.optical_flow_path)
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(test_fires, self.metadata['fire_to_images'], (-2400,2400), self.series_length, self.add_base_flow, self.optical_flow_path)
            
            # Create train/val/test split of Images
            # Only remove images from omit_images_list if not masking
            omit_images_list = None if self.mask_omit_images else self.omit_images_list
            
            self.train_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], train_fires, omit_images_list)
            self.val_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], val_fires, omit_images_list)
            # Don't omit images from test
            self.test_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], test_fires, None)

            # If logdir is provided, then save train/val/test splits
            if log_dir:
                os.makedirs(log_dir)
                np.savetxt(log_dir+'/train_images.txt', self.train_split, fmt='%s')
                np.savetxt(log_dir+'/val_images.txt', self.val_split, fmt='%s')
                np.savetxt(log_dir+'/test_images.txt', self.test_split, fmt='%s')

        if self.tile_label_stats is not None:
            expected_num_tiles = util_fns.calculate_num_tiles(
                self.resize_dimensions,
                self.crop_height,
                self.tile_dimensions,
                self.tile_overlap)
            expected_num_tiles = expected_num_tiles[0] * expected_num_tiles[1]
            invalid_stats = [
                image_name for image_name, counts in self.tile_label_stats.items()
                if len(counts) != expected_num_tiles
            ]
            if invalid_stats:
                raise ValueError(
                    f'Precomputed labels do not match the configured grid of '
                    f'{expected_num_tiles} tiles. Invalid entries: {invalid_stats[:5]}'
                )

            train_before = len(self.train_split)
            missing_positive_stats = [
                image_name for image_name in self.train_split
                if util_fns.get_ground_truth_label(image_name) == 1
                and image_name not in self.tile_label_stats
            ]
            if missing_positive_stats:
                missing_set = set(missing_positive_stats)
                self.train_split = [
                    image_name for image_name in self.train_split
                    if image_name not in missing_set
                ]

            covered_positives = sum(
                util_fns.get_ground_truth_label(image_name) == 1
                for image_name in self.train_split
            )
            negatives = len(self.train_split) - covered_positives
            print('Precomputed tile-label summary:')
            print(f'- grid: {expected_num_tiles} tiles')
            print(f'- train images: {len(self.train_split)} ({covered_positives} positive, {negatives} negative)')
            print(f'- removed positive images without tile labels: {train_before - len(self.train_split)}')
        
        self.has_setup = True
        print("Setting Up Data Complete.")
    
    
    def train_dataloader(self):
        train_dataset = DynamicDataloader(raw_data_path=self.raw_data_path,
                                          labels_path=self.labels_path, 
                                          tile_label_stats=self.tile_label_stats,
                                          optical_flow_path=self.optical_flow_path,
                                          split_name='train',
                                          
                                          metadata=self.metadata, 
                                          data_split=self.train_split,
                                          omit_images_list=self.omit_images_list if self.mask_omit_images else None,
                                          is_object_detection=self.is_object_detection,
                                          is_maskrcnn=self.is_maskrcnn,
                                          is_background_removal=self.is_background_removal,
                                          
                                          original_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                          
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          pre_tile=self.pre_tile,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=self.num_tile_samples,
                                          
                                          flip_augment=self.flip_augment,
                                          resize_crop_augment=self.resize_crop_augment,
                                          blur_augment=self.blur_augment,
                                          color_augment=self.color_augment,
                                          brightness_contrast_augment=self.brightness_contrast_augment)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers,
                                  pin_memory=True, 
                                  shuffle=True,
                                  collate_fn=util_fns.default_collate if self.is_object_detection else None)
        return train_loader

    def val_dataloader(self):
        val_dataset = DynamicDataloader(raw_data_path=self.raw_data_path, 
                                          labels_path=self.labels_path, 
                                          tile_label_stats=self.tile_label_stats,
                                          optical_flow_path=self.optical_flow_path,
                                          split_name='val',
                                        
                                          metadata=self.metadata,
                                          data_split=self.val_split,
                                          omit_images_list=self.omit_images_list if self.mask_omit_images else None,
                                          is_object_detection=self.is_object_detection,
                                          is_maskrcnn=self.is_maskrcnn,
                                          is_background_removal=self.is_background_removal,
                                        
                                          original_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                        
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          pre_tile=self.pre_tile,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=self.num_tile_samples,
                                        
                                          flip_augment=False,
                                          resize_crop_augment=False,
                                          blur_augment=False,
                                          color_augment=False,
                                          brightness_contrast_augment=False)
        val_loader = DataLoader(val_dataset, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_workers,
                                pin_memory=True,
                                collate_fn=util_fns.default_collate if self.is_object_detection else None)
        return val_loader

    def test_dataloader(self):
        test_dataset = DynamicDataloader(raw_data_path=self.raw_data_path, 
                                          labels_path=self.labels_path, 
                                          tile_label_stats=self.tile_label_stats,
                                          optical_flow_path=self.optical_flow_path,
                                          split_name='test',
                                         
                                          metadata=self.metadata,
                                          data_split=self.test_split,
                                          omit_images_list=None,
                                          is_object_detection=self.is_object_detection,
                                          is_maskrcnn=self.is_maskrcnn,
                                          is_background_removal=self.is_background_removal,
                                         
                                          original_dimensions=self.original_dimensions,
                                          resize_dimensions=self.resize_dimensions,
                                          crop_height=self.crop_height,
                                         
                                          tile_dimensions=self.tile_dimensions,
                                          tile_overlap=self.tile_overlap,
                                          pre_tile=self.pre_tile,
                                          smoke_threshold=self.smoke_threshold,
                                          num_tile_samples=0,
                                         
                                          flip_augment=False,
                                          resize_crop_augment=False,
                                          blur_augment=False,
                                          color_augment=False,
                                          brightness_contrast_augment=False)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.batch_size, 
                                 num_workers=self.num_workers,
                                 pin_memory=True,
                                 collate_fn=util_fns.default_collate if self.is_object_detection else None)
        return test_loader
    
    
    
#####################
## Dataloader
#####################
    
class DynamicDataloader(Dataset):
    def __init__(self, 
                 raw_data_path=None,
                 labels_path=None, 
                 tile_label_stats=None,
                 optical_flow_path=None,
                 split_name='train',
                 
                 metadata=None,
                 data_split=None, 
                 omit_images_list=None,
                 is_object_detection=False,
                 is_maskrcnn=False,
                 is_background_removal=False,
                 
                 original_dimensions = (1536, 2016),
                 resize_dimensions = (1536, 2016),
                 crop_height = 1120,
                 tile_dimensions = (224,224), 
                 tile_overlap = 0,
                 pre_tile = True,
                 smoke_threshold = 250,
                 num_tile_samples = 0,
                 
                 flip_augment = True,
                 resize_crop_augment = True,
                 blur_augment = True,
                 color_augment = True,
                 brightness_contrast_augment = True):
        
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.tile_label_stats = tile_label_stats
        self.optical_flow_path = optical_flow_path
        self.split_name = split_name
        
        self.metadata = metadata
        self.data_split = data_split
        self.omit_images_list = omit_images_list
        self.is_object_detection = is_object_detection
        self.is_maskrcnn = is_maskrcnn
        self.is_background_removal = is_background_removal
        
        self.original_dimensions = original_dimensions
        self.resize_dimensions = resize_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.tile_overlap = tile_overlap
        self.pre_tile = pre_tile
        self.smoke_threshold = smoke_threshold
        self.num_tile_samples = num_tile_samples
        
        self.flip_augment = flip_augment
        self.resize_crop_augment = resize_crop_augment
        self.blur_augment = blur_augment
        self.color_augment = color_augment
        self.brightness_contrast_augment = brightness_contrast_augment
        
        self.num_tiles_height, self.num_tiles_width = (1,1) if self.tile_dimensions is None else util_fns.calculate_num_tiles(resize_dimensions, crop_height, tile_dimensions, tile_overlap)

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        image_name = self.data_split[idx]
        series_length = len(self.metadata['image_series'][image_name])
                
        data_augmentations = util_fns.DataAugmentations(self.original_dimensions,
                                                        self.resize_dimensions,
                                                        self.tile_dimensions,
                                                        self.crop_height,
                                                        
                                                        self.flip_augment, 
                                                        self.resize_crop_augment, 
                                                        self.blur_augment, 
                                                        self.color_augment, 
                                                        self.brightness_contrast_augment)
        
        ### Load Images ###
        x = []
        
        for file_name in self.metadata['image_series'][image_name]:
            # Load image
            # img.shape = [height, width, num_channels]
            img_path = Path(self.raw_data_path) / f"{file_name}.jpg"
            if not img_path.exists():
                raise FileNotFoundError(f"No se encontró la imagen: {img_path}")
            img = cv2.imread(str(img_path))
                
            # Apply data augmentations
            # img.shape = [crop_height, resize_dimensions[1], num_channels]
            img = data_augmentations(img, is_labels=False)
                
            # Tile image
            # img.shape = [num_tiles, tile_height, tile_width, num_channels]
            if self.pre_tile and not self.is_object_detection:
                img = util_fns.tile_image(img, self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)
            
            # Rescale and normalize
            if self.is_object_detection:
                img = img / 255 # torchvision object detection expects input [0,1]
            else:
                img = util_fns.normalize_image(img)

            x.append(img)
        
        if self.pre_tile and not self.is_object_detection:
            # x.shape = [num_tiles, series_length, num_channels, tile_height, tile_width]
            x = np.transpose(np.stack(x), (1, 0, 4, 2, 3))
        else:
            # x.shape = [series_length, num_channels, height, width]
            x = np.transpose(np.stack(x), (0, 3, 1, 2))
           
        ### Load Labels ###
        ground_truth_label = util_fns.get_ground_truth_label(image_name)
        tiled_labels = np.array([0])
        bbox_labels = []
        omit_mask = False
        
        ### Load Tile Labels ###
        if self.is_maskrcnn or not self.is_object_detection:
            has_precomputed_labels = (
                not self.is_object_detection
                and self.tile_label_stats is not None
                and image_name in self.tile_label_stats
            )
            if has_precomputed_labels:
                tile_counts = np.asarray(self.tile_label_stats[image_name])
                tiled_labels = (tile_counts > self.smoke_threshold).astype(float)

                # Images and tiles are flattened row-major, so a horizontal flip
                # reverses the tile columns while preserving row order.
                if data_augmentations.should_flip:
                    tiled_labels = tiled_labels.reshape(
                        self.num_tiles_height, self.num_tiles_width
                    )[:, ::-1].reshape(-1).copy()

                omit_mask = True
            else:
                label_path = None if self.labels_path is None else Path(self.labels_path) / f'{image_name}.npy'

            if not has_precomputed_labels and label_path is not None and label_path.exists():
                # Repeat similar steps to images
                labels = np.load(label_path)

                labels = data_augmentations(labels, is_labels=True)
            elif not has_precomputed_labels:
                # Load all 0s
                # labels.shape = [height, width]
                labels = np.zeros((self.crop_height, self.resize_dimensions[1])).astype(float) 

            if not self.is_object_detection and not has_precomputed_labels:
                # Tile labels
                tiled_labels = util_fns.tile_labels(labels, self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)

                # Generate binary labels basaed on smoke_threshold
                # labels.shape = [num_tiles]
                tiled_labels = (tiled_labels.sum(axis=(1,2)) > self.smoke_threshold).astype(float)

                # Random upsampling
                if self.num_tile_samples > 0:
                    # WARNING: Assumes that there are no labels with all 0s. Use --time-range-min 0
                    x, tiled_labels = util_fns.randomly_sample_tiles(x, tiled_labels, self.num_tile_samples)
            
                # Determine if tile predictions should be masked
                omit_mask = False if (self.omit_images_list is not None and (image_name in self.omit_images_list or util_fns.get_fire_name(image_name) in self.metadata['unlabeled_fires'])) else True
        
        ### Load Object Detection Labels ###
        if self.is_object_detection:
            # Loop through each image in series
            for image in x:
                bbox_label = {}
                
                if image_name in self.metadata['bbox_labels']: 
                    # Append real positive image data
                    bboxes = data_augmentations.process_bboxes(self.metadata['bbox_labels'][image_name])
                    bbox_label['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32)
                    bbox_label['labels'] = torch.as_tensor([1]*len(self.metadata['bbox_labels'][image_name]), dtype=torch.int64)
                    if self.is_maskrcnn:
                        bbox_label['masks'] = torch.as_tensor(np.expand_dims(labels, 0), dtype=torch.uint8)
                else:
                    # Use negative data
                    # Source: https://github.com/pytorch/vision/releases/tag/v0.6.0
                    bbox_label = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                                  "labels": torch.zeros(0, dtype=torch.int64),
                                  "area": torch.zeros(0, dtype=torch.float32)}
                    if self.is_maskrcnn:
                        bbox_label['masks'] = torch.zeros((0, self.crop_height, self.resize_dimensions[1]), dtype=torch.uint8)
                    
                bbox_labels.append(bbox_label)
        
        ### Load Optical Flow ###
        if self.optical_flow_path is not None:
            flow_imgs = []
            
            for file_name in self.metadata['image_series'][image_name]:
                # Load flow
                # img.shape = [height, width, num_channels]
                img = np.load(self.optical_flow_path+'/'+file_name+'.npy')

                # Apply data augmentations
                # img.shape = [crop_height, resize_dimensions[1], num_channels]
                img = data_augmentations(img, is_labels=True)
                
                # Tile image
                # img.shape = [num_tiles, tile_height, tile_width, num_channels]
                if self.pre_tile and not self.is_object_detection:
                    if self.is_background_removal:
                        img = util_fns.tile_labels(img, self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)
                        img = np.expand_dims(img, 3)
                    else:
                        img = util_fns.tile_image(img, self.num_tiles_height, self.num_tiles_width, self.resize_dimensions, self.tile_dimensions, self.tile_overlap)

                # Rescale and normalize
                if self.is_object_detection:
                    img = img / 255 # torchvision object detection expects input [0,1]
                else:
                    img = util_fns.normalize_image(img)

                flow_imgs.append(img)

            if self.pre_tile and not self.is_object_detection:
                # flow_imgs.shape = [num_tiles, series_length, num_channels, tile_height, tile_width]
                flow_imgs = np.transpose(np.stack(flow_imgs), (1, 0, 4, 2, 3))
                x = np.concatenate([x, flow_imgs], axis=2)
            else:
                # flow_imgs.shape = [series_length, num_channels, height, width]
                flow_imgs = np.transpose(np.stack(flow_imgs), (0, 3, 1, 2))
                x = np.concatenate([x, flow_imgs], axis=1)
        
        # Address ambiguous batch_size warning
        image_name = image_name if self.split_name == 'test' else 0
        
        return image_name, x, tiled_labels, bbox_labels, ground_truth_label, omit_mask
