import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class To5vStrokes():
    def __init__(self, max_len=200):
        self.max_len = max_len

    def __call__(self, stroke):
        """
        Converts from stroke-3(numpy array) to stroke-5(torch tensor) format and pads to given length.
        (But does not insert special start token).
        """
        result = np.zeros((self.max_len, 5), dtype=float)
        l = len(stroke)
        assert l <= self.max_len
        result[0:l, 0:2] = stroke[:, 0:2]
        result[0:l, 3] = stroke[:, 2]
        result[0:l, 2] = 1 - result[0:l, 3]
        result[l:, 4] = 1
        return torch.from_numpy(result).float().to(device)


class V5Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, pre_scaling=True):
        super().__init__()
        self.transform = transform
        self.data = np.load(data_path, encoding='bytes', allow_pickle=True)
        scale = self.scaling_factor()
        self.data = np.array(list(
            map(lambda x: self.scale_stroke(x, scale), self.data)))
        if transform is not None:
            self.data = np.array(list(map(transform, self.data)))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], 0

    def scaling_factor(self):
        data = np.concatenate([S for S in self.data])
        return np.std(data[:, 0:2])

    def scale_stroke(self, x, scale):
        x = np.float32(x)
        x[:, 0:2] /= scale
        return x
