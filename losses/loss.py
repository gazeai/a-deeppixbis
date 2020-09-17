import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# loss definitions

class AngularLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type='binary-angularloss', m=None, eps=1e-7):
        super(AngularLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['binary-angularloss']
        if loss_type == 'binary-angularloss':
            self.m = 0.5 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.fc = nn.Linear(in_features, out_features, bias=False)
        
    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        wf = self.fc(x)
        theta = wf
        theta = torch.clamp(theta, -1.+self.eps, 1-self.eps)
        theta = torch.acos(theta)
        pos = nn.Sigmoid()(torch.cos(theta+self.m))
        neg = nn.Sigmoid()(torch.cos(theta))
        labels = torch.unsqueeze(labels, 1)
        L = labels*torch.log(pos) + (1-labels)*torch.log(1-neg)
        L = torch.mean(L)
#         print(f"---------------------sigmoid----------------------------------")
#         print(nn.Sigmoid()(wf)[0], labels[0])
#         print(nn.Sigmoid()(wf)[1], labels[1])
#         print(nn.Sigmoid()(wf)[2], labels[2])
#         print(nn.Sigmoid()(wf)[3], labels[3])
#         print(nn.Sigmoid()(wf)[4], labels[4])
#         print(nn.Sigmoid()(wf)[5], labels[5])
        return -L, nn.Sigmoid()(wf)
    
    
class AngularConvLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type='binary-conv-angularloss', m=None, eps=1e-7):
        super(AngularConvLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['binary-conv-angularloss']
        if loss_type == 'binary-conv-angularloss':
            self.m = 0.5 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.enc = nn.Conv2d(in_features, out_features, kernel_size=1, padding=0, bias=False)
        self.attn = nn.Parameter(torch.empty(14*14).fill_(1), requires_grad=True)
        
    def forward(self, x, labels):
        for W in self.enc.parameters():
            W = F.normalize(W, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        wf = self.enc(x)
        
        theta = wf
        theta = theta.reshape(theta.shape[0], 196)
        theta = torch.clamp(theta, -1.+self.eps, 1-self.eps)
        theta = torch.acos(theta)
        labels = labels.reshape(labels.shape[0], 196)
        pos = nn.Sigmoid()(torch.cos(theta+self.m))
        neg = nn.Sigmoid()(torch.cos(theta))
        
        L = labels*torch.log(pos) + (1-labels)*torch.log(1-neg)
        atten = nn.functional.softmax(self.attn)
        L = L * atten
        L = torch.sum(L)

#         L = torch.mean(L)
        
#         print(f"+++++++++++++++++++feature map++++++++++++++++++++++{L}++++{torch.sum(atten)}")
        
#         print(nn.Sigmoid()(torch.mean(wf[0])).data.item(), torch.mean(labels[0]).data.item())
#         print(nn.Sigmoid()(torch.mean(wf[1])).data.item(), torch.mean(labels[1]).data.item())
#         print(nn.Sigmoid()(torch.mean(wf[2])).data.item(), torch.mean(labels[2]).data.item())
#         print(nn.Sigmoid()(torch.mean(wf[3])).data.item(), torch.mean(labels[3]).data.item())
#         print(nn.Sigmoid()(torch.mean(wf[4])).data.item(), torch.mean(labels[4]).data.item())
#         print(nn.Sigmoid()(torch.mean(wf[5])).data.item(), torch.mean(labels[5]).data.item())
        return -L, nn.Sigmoid()(wf)
    
    
def compute_loss(network, img, labels, device):
    """
    Compute the losses, given the network, data and labels and
    device in which the computation will be performed.
    """
    labelsv_pixel = Variable(labels['pixel_mask'].to(device))
    labelsv_binary = Variable(labels['binary_target'].to(device))
    angular_loss, angular_loss_conv, patch_map, p = network(img.to(device), labelsv_binary, labelsv_pixel)
    beta = 0.5

    loss = beta*angular_loss + (1-beta)*angular_loss_conv
    return loss