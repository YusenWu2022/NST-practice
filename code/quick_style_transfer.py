import copy
import torch
import tkinter as tk
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms


class ResBlock(nn.Module):

    def __init__(self, c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(c),

        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layer(x)+x)


class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 9, 1, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


def gram_func(f_map):
    n, c, h, w = f_map.shape
    f_map = f_map.reshape(n, c, h * w)
    gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
    return gram_matrix


loader = transforms.Compose([transforms.ToTensor()])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_name):
    img = Image.open(image_name)
    #img = img.resize((imsize, imsize), Image.ANTIALIAS)
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        #cnn = models.vgg19(pretrained=True).features.to(device).eval()
        a = models.vgg16(pretrained=True).features.to(device).eval()
        self.layer1 = a[:4]
        self.layer2 = a[4:9]
        self.layer3 = a[9:16]
        self.layer4 = a[16:23]
#-----------------------------putout 4 layers feature picture------------------------------------------

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


image_style = load_image(
    'D:\python\pythontest\png_form_images\jpg\image_style1.jpg')
vgg16 = VGG16()
t_net = TransNet()
g_net=vgg16
#g_net.load_state_dict(torch.load('fst.pth'))  #putout the prediction
optimizer = torch.optim.Adam(g_net.parameters())
loss_func = nn.MSELoss()
data_set = COCODataSet()#load data
batch_size = 4
data_loader = DataLoader(data_set, batch_size, True, drop_last=True)

#---------------------calculate gram matrix------------------------------------------
s1, s2, s3, s4 = vgg16(image_style)
s1 = gram_func(s1).detach().expand(batch_size, s1.shape[1], s1.shape[1])
s2 = gram_func(s2).detach().expand(batch_size, s2.shape[1], s2.shape[1])
s3 = gram_func(s3).detach().expand(batch_size, s3.shape[1], s3.shape[1])
s4 = gram_func(s4).detach().expand(batch_size, s4.shape[1], s4.shape[1])
j = 0
while True:
    for i, image in enumerate(data_loader):
        #give out image and calculate the loss
        image_c = image.cuda()
        image_g = t_net(image_c)
        out1, out2, out3, out4 = vgg16(image_g)
        # loss = loss_func(image_g, image_c)
        #style loss
        loss_s1 = loss_func(gram_func(out1), s1)
        loss_s2 = loss_func(gram_func(out2), s2)
        loss_s3 = loss_func(gram_func(out3), s3)
        loss_s4 = loss_func(gram_func(out4), s4)
        loss_s = loss_s1+loss_s2+loss_s3+loss_s4

        # content loss
        c1, c2, c3, c4 = vgg16(image_c)

        # loss_c1 = loss_func(out1, c1.detach())
        loss_c2 = loss_func(out2, c2.detach())
        # loss_c3 = loss_func(out3, c3.detach())
        # loss_c4 = loss_func(out4, c4.detach())

        # total loss
        loss = loss_c2 + 0.000008 * loss_s

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(j, i, loss.item(), loss_c2.item(), loss_s.item())
        if i % 100 == 0:
            torch.save(g_net.state_dict(), 'fst.pth')
            save_image([image_g[0], image_c[0]], f'D:/data/{i}.jpg', padding=0, normalize=True,
                       range=(0, 1))
            j += 1

# pretrained=True   (use pre_trained model)
# net = models.resnet34(pretrained=False)
#------------------------------------load model---------------------------------------------
pthfile = 'D:\python_cv\pth'
net.load_state_dict(torch.load(pthfile))
print(net)
test = load_image("")
test_trans = net(test)
printf(test_trans)
