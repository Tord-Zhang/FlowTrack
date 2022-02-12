import cv2
import random
import numpy as np
import torch

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data



class RandomCrop:
    def __call__(self, data):
        img1, img2, box1, box2 = data
        h, w, _ = img1.shape
        ds = min(h, w)

        # x = (w - ds) // 2
        # y = (h - ds) // 2
        x = random.randint(0, (w - ds) // 2)
        y = random.randint(0, (h - ds) // 2)

        #make sure that the target is not croped out
        x = min(box1[0], x)
        y = min(box1[1], y)

        img1 = img1[y : y + ds, x : x + ds]
        img2 = img2[y : y + ds, x : x + ds]

        box1[0] -= x
        box1[1] -= y
        if box1[0] < 0:
            box1[2] -= abs(box1[0])
            box1[0] = 0
        if box1[1] < 0:
            box1[3] -= abs(box1[1])
            box1[1] = 0

        box1[0] = max(0, box1[0])
        box1[1] = max(0, box1[1])
        box1[2] = min(ds-box1[0], box1[2])
        box1[3] = min(ds-box1[1], box1[3])

        box2[0] -= x
        box2[1] -= y
        if box2[0] < 0:
            box2[2] -= abs(box2[0])
            box2[0] = 0
        if box2[1] < 0:
            box2[3] -= abs(box2[1])
            box2[1] = 0

        box2[0] = max(0, box2[0])
        box2[1] = max(0, box2[1])
        box2[2] = min(ds-box2[0], box2[2])
        box2[3] = min(ds-box2[1], box2[3])
        
        return (img1, img2, box1, box2) 


class Resize:
    def __init__(self, length, multiple=1):
        self.length = length
        self.multiple = multiple

    def __call__(self, data):
        img1, img2, box1, box2 = data
        h, w, _ = img1.shape
        scale = self.length / float(min(h, w))

        dh = int(round(h * scale / self.multiple) * self.multiple)
        dw = int(round(w * scale / self.multiple) * self.multiple)
        
        img1 = cv2.resize(
            img1,
            (dw, dh),
            interpolation=cv2.INTER_LINEAR_EXACT,
        )
        img2 = cv2.resize(
            img2,
            (dw, dh),
            interpolation=cv2.INTER_NEAREST,
        )

        box1 = np.round(box1 * scale).astype(np.int)
        box2 = np.round(box2 * scale).astype(np.int)

        return (img1, img2, box1, box2)

class ToTensor(object):

    def __call__(self, data):
        img1, img2, box1, box2 = data
        img1 = np.transpose(img1, (2, 0, 1)).astype(np.float32) / 255.
        img2 = np.transpose(img2, (2, 0, 1)).astype(np.float32) / 255.

        return (torch.from_numpy(img1), torch.from_numpy(img2), torch.from_numpy(box1), torch.from_numpy(box2))

if __name__ == "__main__":
    transforms = Compose(
        [
        CenterCrop(),
        Resize(length=512, multiple=1),
        ]
    )
    img = cv2.imread("data/ILSVRC2015_VID/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00017008/000000.JPEG")
    box = np.array([112, 280, 312, 480])
    print(img.shape)
    data = transforms((img, img, box, box))
    print(data[1].shape, data[2])