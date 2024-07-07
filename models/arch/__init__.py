import os
from models.arch.DURRNet import *

def durrnet(in_channels, out_channels, **kwargs):
    return DURRNet(in_channels, **kwargs)


if __name__ == '__main__':
    x = torch.ones(1, 1475, 256, 256)
    net = durrnet(1475, 3)
    print(net)
    url = "./tmp.pth"
    torch.save(net.state_dict(), url)
    print('\n', os.path.getsize(url) / (1024 * 1024), 'MB')
    l, r = net(x)
