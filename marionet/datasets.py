"""Dataset classes."""
import os

import numpy as np
from PIL import Image
import scipy
import scipy.cluster
import torch as th
from torchvision.transforms.functional import to_tensor, crop


class AnimationDataset(th.utils.data.Dataset):
    def __init__(self, data, canvas_size=128):
        self.canvas_size = canvas_size
        self.files = [os.path.join(data, f)
                      for f in os.listdir(data)
                      if f.endswith('png') or f.endswith('jpg')
                      or f.endswith('jpeg')]
        self.files = sorted(
            self.files,
            key=lambda f: int(''.join(x for x in os.path.basename(f)
                                      if x.isdigit())))

        colors = []
        for f in np.random.choice(self.files, size=min(len(self.files), 100),
                                  replace=False):
            im = Image.open(f).convert('RGB')
            w, h = im.size
            a = min(w, h) / self.canvas_size
            im = im.resize((int(w/a), int(h/a)), Image.LANCZOS)
            colors.append(np.asarray(im).reshape(-1, 3).astype(float))

        colors = np.vstack(colors)
        codes, _ = scipy.cluster.vq.kmeans(colors, 5)
        vecs, _ = scipy.cluster.vq.vq(colors, codes)
        counts, _ = scipy.histogram(vecs, len(codes))
        self.bg = codes[scipy.argmax(counts)].astype(np.float32) / 255

    def __repr__(self):
        return "AnimationDataset | {} entries".format(len(self))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = Image.open(self.files[idx]).convert('RGB')
        w, h = im.size
        a = min(w, h) / self.canvas_size
        im = im.resize((int(w/a), int(h/a)), Image.LANCZOS)
        w, h = im.size
        im = im.crop(((w-self.canvas_size) // 2, (h-self.canvas_size) // 2,
                      (w-self.canvas_size) // 2 + self.canvas_size,
                      (h-self.canvas_size) // 2 + self.canvas_size))
        im = to_tensor(im)

        return {
            'im': im,
            'fname': self.files[idx]
        }
