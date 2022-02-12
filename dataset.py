from __future__ import absolute_import, print_function

import os
import glob
import six
import numpy as np
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import cv2


class ImageNetVID(object):
    r"""`ImageNet Video Image Detection (VID) <https://image-net.org/challenges/LSVRC/2015/#vid>`_ Dataset.
    Publication:
        ``ImageNet Large Scale Visual Recognition Challenge``, O. Russakovsky,
            J. deng, H. Su, etc. IJCV, 2015.
    
    Args:
        root_dir (string): Root directory of dataset where ``Data``, and
            ``Annotation`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or (``train``, ``val``)
            subset(s) of ImageNet-VID. Default is a tuple (``train``, ``val``).
        cache_dir (string, optional): Directory for caching the paths and annotations
            for speeding up loading. Default is ``cache/imagenet_vid``.
    """
    def __init__(self, root_dir, subset=('train', 'val'),
                 cache_dir='cache/imagenet_vid'):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        if isinstance(subset, str):
            assert subset in ['train', 'val']
            self.subset = [subset]
        elif isinstance(subset, (list, tuple)):
            assert all([s in ['train', 'val'] for s in subset])
            self.subset = subset
        else:
            raise Exception('Unknown subset')
        
        # cache filenames and annotations to speed up training
        self.seq_dict = self._cache_meta()
        self.seq_names = [n for n in self.seq_dict]

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            seq_name = index
        else:
            seq_name = self.seq_names[index]

        seq_dir, frames, anno_file = self.seq_dict[seq_name]
        img_files = [os.path.join(
            seq_dir, '%06d.JPEG' % f) for f in frames]
        anno = np.loadtxt(anno_file, delimiter=',')

        return img_files, anno

    def __len__(self):
        return len(self.seq_dict)

    def _cache_meta(self):
        cache_file = os.path.join(self.cache_dir, 'seq_dict.json')
        if os.path.isfile(cache_file):
            print('Dataset already cached.')
            with open(cache_file) as f:
                seq_dict = json.load(f, object_pairs_hook=OrderedDict)
            return seq_dict
        
        # image and annotation paths
        print('Gather sequence paths...')
        seq_dirs = []
        anno_dirs = []
        if 'train' in self.subset:
            seq_dirs_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/train/ILSVRC*/ILSVRC*')))
            anno_dirs_ = [os.path.join(
                self.root_dir, 'Annotations/VID/train',
                *s.split('/')[-2:]) for s in seq_dirs_]
            seq_dirs += seq_dirs_
            anno_dirs += anno_dirs_
        if 'val' in self.subset:
            seq_dirs_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/val/ILSVRC2015_val_*')))
            anno_dirs_ = [os.path.join(
                self.root_dir, 'Annotations/VID/val',
                s.split('/')[-1]) for s in seq_dirs_]
            seq_dirs += seq_dirs_
            anno_dirs += anno_dirs_
        seq_names = [os.path.basename(s) for s in seq_dirs]

        # cache paths and annotations
        print('Caching annotations to %s, ' % self.cache_dir + \
            'it may take a few minutes...')
        seq_dict = OrderedDict()
        cache_anno_dir = os.path.join(self.cache_dir, 'anno')
        if not os.path.isdir(cache_anno_dir):
            os.makedirs(cache_anno_dir)

        for s, seq_name in enumerate(seq_names):
            if s % 100 == 0 or s == len(seq_names) - 1:
                print('--Caching sequence %d/%d: %s' % \
                    (s + 1, len(seq_names), seq_name))
            anno_files = sorted(glob.glob(os.path.join(
                anno_dirs[s], '*.xml')))
            objects = [ET.ElementTree(file=f).findall('object')
                       for f in anno_files]
            
            # find all track ids
            track_ids, counts = np.unique([
                obj.find('trackid').text for group in objects
                for obj in group], return_counts=True)
            
            # fetch paths and annotations for each track id
            for t, track_id in enumerate(track_ids):
                if counts[t] < 2:
                    continue
                frames = []
                anno = []
                for f, group in enumerate(objects):
                    for obj in group:
                        if not obj.find('trackid').text == track_id:
                            continue
                        frames.append(f)
                        anno.append([
                            int(obj.find('bndbox/xmin').text),
                            int(obj.find('bndbox/ymin').text),
                            int(obj.find('bndbox/xmax').text),
                            int(obj.find('bndbox/ymax').text)])
                anno = np.array(anno, dtype=int)
                anno[:, 2:] -= anno[:, :2] - 1

                # store annotations
                key = '%s.%d' % (seq_name, int(track_id))
                cache_anno_file = os.path.join(cache_anno_dir, key + '.txt')
                np.savetxt(cache_anno_file, anno, fmt='%d', delimiter=',')

                # store paths
                seq_dict.update([(key, [
                    seq_dirs[s], frames, cache_anno_file])])
        
        # store seq_dict
        with open(cache_file, 'w') as f:
            json.dump(seq_dict, f)

        return seq_dict



class FlowTrackDataset(Dataset):
    def __init__(self, dataset_root, transforms, crop_size=384,):
        self.imagenet_vid = ImageNetVID(dataset_root)
        self.dataset_len = len(self.imagenet_vid)
        self.transforms = transforms


    def __len__(self,):
        return self.dataset_len

    def __getitem__(self, index):
        #get a seq
        random_idx = random.randint(0, self.dataset_len - 1)
        img_files, anno = self.imagenet_vid[random_idx]
        num_imgs = len(img_files)

        # take two frames from the seq
        interval = random.randint(1, 20)
        frame1_idx = random.randint(0, num_imgs - interval - 1)
        frame1_path = img_files[frame1_idx]
        box1 = np.array(anno[frame1_idx]).astype(np.int)

        frame2_idx = frame1_idx + interval
        frame2_path = img_files[frame2_idx]
        box2 = np.array(anno[frame2_idx]).astype(np.int)

        img1 = cv2.imread(frame1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(frame2_path, cv2.IMREAD_COLOR)
        # import copy
        # img1_vis = copy.deepcopy(img1)
        # cv2.rectangle(img1_vis, (box1[0], box1[1]), (box1[0]+box1[2], box1[1]+box1[3]), (0,0,255), 3)
        # cv2.imwrite("test1_before.png", img1_vis)

        # img2_vis = copy.deepcopy(img2)
        # cv2.rectangle(img2_vis, (box2[0], box2[1]), (box2[0]+box2[2], box2[1]+box2[3]), (0,0,255), 3)
        # cv2.imwrite("test2_before.png", img2_vis)

        return self.transforms((img1, img2, box1, box2))
    


if __name__ == "__main__":
    from dataset_utils import Compose, RandomCrop, Resize

    # corr = [615,0,664,306]
    # img = cv2.imread("data/ILSVRC2015_VID/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00017006/000000.JPEG")
    # cv2.rectangle(img, (corr[0], corr[1]), (corr[0]+corr[2], corr[1]+corr[3]), (0,0,255), 3)
    # cv2.imwrite("test.png", img)
    # exit()

    transforms = Compose(
        [
        RandomCrop(),
        # Resize(length=384, multiple=1)
        ]
    )

    data = ImageNetVID("data/ILSVRC2015_VID/ILSVRC2015")
    img_files, anno = data[0]
    
    dataset = FlowTrackDataset("data/ILSVRC2015_VID/ILSVRC2015", transforms)

    img1, img2, box1, box2 = dataset[0]
    
    print(img1.shape)
    print(box1)
    cv2.rectangle(img1, (box1[0], box1[1]), (box1[0]+box1[2], box1[1]+box1[3]), (0,0,255), 3)
    cv2.imwrite("test1.png", img1)

    cv2.rectangle(img2, (box2[0], box2[1]), (box2[0]+box2[2], box2[1]+box2[3]), (0,0,255), 3)
    cv2.imwrite("test2.png", img2)
    # img = cv2.imread("data/ILSVRC2015_VID/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00017008/000000.JPEG")
    # print(img.shape)


#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