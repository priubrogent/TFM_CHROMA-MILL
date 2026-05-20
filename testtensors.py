import torch

I = torch.tensor([100, 180, 200, 200, 200, 200, 200, 200, 200, 200, 200])

img = torch.rand(I.shape[0], 3, 224, 224)

channelI = torch.ones(I.shape[0], 1, img.shape[2], img.shape[3])*I[:, None, None, None]


print(channelI[:, 0, 0, 0])