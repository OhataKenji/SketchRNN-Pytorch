import torch
import matplotlib.pyplot as plt
import PIL
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def strokes2rgb(S):
    plt.axis('equal')
    S = S.clone().detach().cpu()
    N = S.shape[0]

    p2list = [-1]
    prev = torch.tensor([0, 0], device='cpu', dtype=torch.float)
    for i in range(N):
        S[i, 0, 0:2] += prev
        prev = S[i, 0, 0:2]
        if S[i, 0, 3] == 1:
            p2list.append(i)
    p2list.append(N-1)

    for i in range(len(p2list)-1):
        s = p2list[i]
        e = p2list[i+1]
        plt.plot(S[s+1:e+1, 0, 0], -S[s+1:e+1, 0, 1])

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    plt.close("all")
    img_array = np.asarray(pil_image)
    return np.transpose(img_array, (2, 0, 1))
