import torch
import torch.nn as nn


class StyleRandomization(nn.Module): ## same content with randomly interpolated style
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        # print(self.training) # True
        # print('************') 
        if self.training:
            x = x.view(N, C, -1) 
            # print(x.shape) # torch.Size([2, 512, 8160]) 
            # print('***************')
            mean = x.mean(-1, keepdim=True) 
            # print(mean.shape) # torch.Size([2, 512, 1])
            # print(mean[torch.randperm(2)].shape) # # torch.Size([2, 512, 1]) 
            # print('**********') 
            var = x.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()
            
            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)
            if x.is_cuda:
                alpha = alpha.cuda()
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]  ## so this mean[idx_swap] deals with the zeroth dim and it swaps the means in Batch dimension for having different mean configuration in the Batch size.
            var = alpha * var + (1 - alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x


class ContentRandomization(nn.Module):  ## different content with same style
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()

        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            
            x = (x - mean) / (var + self.eps).sqrt()  ## z score normalisation 
            
            idx_swap = torch.randperm(N)
            x = x[idx_swap].detach()  ## this is a different input where, back prop isn't req...and now this is the content

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x
