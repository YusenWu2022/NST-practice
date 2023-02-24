from __future__ import print_function
import copy
import torch
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
# ------some important value set------------------------------------------
content_origin_lenth = 0
content_origin_width = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_turns = 0
flag=False
imsize = 512
# -----------set file path and size---------------------------------------------
windows = tk.Tk()
windows.title("insert wanted picture path and size")
windows.geometry("200x300")

image = Image.open("D:\python\pythontest\photos\sailboat_transformed.jpg")
photo = ImageTk.PhotoImage(image)
theLabel = tk.Label(windows, text="choose your wish", justify=tk.LEFT,
                    image=photo, compound=tk.CENTER, font=("华文行楷", 20), fg="black")
theLabel.pack()
style_path = 'D:\python\pythontest\photos\style.jpg'
ontent_path = "D:\python\pythontest\photos\sailboat.jpg"
imsize_set = tk.StringVar()
imsize_set.set("512")
style_path_set = tk.StringVar()
style_path_set.set("D:\python\pythontest\photos\style.jpg")
content_path_set = tk.StringVar()
content_path_set.set("D:\python\pythontest\photos\sailboat.jpg")
output_path_set = tk.StringVar()
output_path_set.set("D:\\python_cv\\result\\result.jpg")
size = tk.Entry(windows, show=None, textvariable=imsize_set)
size.pack()
style_path_input = tk.Entry(
    windows, show=None, textvariable=style_path_set)
style_path_input.pack()
content_path_input = tk.Entry(
    windows, show=None, textvariable=content_path_set)
content_path_input.pack()
output_path_input = tk.Entry(
    windows, show=None, textvariable=output_path_set)
output_path_input.pack()
cmd_end = 0


def insert():
    global imsize
    imsize = int(size.get())  # get the input info
    global style_path
    style_path = style_path_input.get()  # get input info
    global content_path
    content_path = content_path_input.get()  # get input info
    img_tmp = Image.open(content_path)
    global content_origin_width
    content_origin_width = img_tmp.width
    global content_origin_lenth
    content_origin_lenth = img_tmp.height
    print(content_origin_width)
    print(content_origin_lenth)
    global output_path
    output_path = output_path_input.get()
    windows.destroy()


cmd = tk.Button(windows, text="confirmed",
                width=15, height=2, command=insert)
cmd.pack()
t = tk.Text(windows, height=2)
t.pack()
windows.mainloop()
# ---------whether separate enfoecement---------------------------------------
windows = tk.Tk()
windows.title("insert wanted picture path and size")
windows.geometry("200x300")
photo = ImageTk.PhotoImage(image)
theLabel = tk.Label(windows, text="want improvement?", justify=tk.LEFT,
                    image=photo, compound=tk.CENTER, font=("华文行楷", 20), fg="black")
theLabel.pack()
def not_improve():
    windows.destroy()

def improve():
    record=0
    flag=True
    def human_segment(net, path, nc=21):
        print("segmenting...")
        img = Image.open(path)
        trf = transforms.Compose([transforms.ToTensor(),  # Transform a pil.image with a value range of [0255] or a numpy.ndarray with a shape of (h, W, c) into a torch.flowedtensor with a shape of [C, h, w] and a value range of [0,1.0]
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Regularize tensor，Normalized_image=(image-mean)/std
                                    std=[0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)  # Returns a new tensor and inserts dimension 1 into the specified position of the input
        out = net(inp)['out']
        image = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        label_colors = np.array([(0, 0, 0),  # 0=background
                                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                (128, 0, 0), (0, 128, 0), (128, 128,
                                                            0), (0, 0, 128), (128, 0, 128),
                                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                (0, 128, 128), (128, 128, 128), (64,
                                                                0, 0), (192, 0, 0), (64, 128, 0),
                                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                (192, 128, 0), (64, 0, 128), (192, 0,
                                                            128), (64, 128, 128), (192, 128, 128),
                                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        # The corresponding category of each pixel is given the corresponding color
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        # this is the color picture of segmenting
        #global rgb = np.stack([r, g, b], axis=2)  # stack up
        rgb = np.stack([r, g, b], axis=2)  ##stack up
        '''
        if record==0 :
        global rgb_content=rgb
        else :
        global rgb_style=rgb
        '''
        plt.imshow(rgb)
        plt.pause(3)
        plt.close() 
        '''
        #取出选中的部分单独处理
        img = cv2.erode(img, rgb)  # 腐蚀 
        img = cv2.dilate(img, rgb)  # 膨胀
        mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #img是腐蚀膨胀完的图片
        ROI = cv2.bitwise_and(mask, oriimg)
        '''
        


    dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
    human_segment(dlab, content_path)
    record+=1
    #human_segment(dlab,style_path)
    windows.destroy()

    
choose1 = tk.Button(windows, text="improve",
                width=15, height=2, command=improve)
choose1.pack()
choose2 = tk.Button(windows, text="not improve",
                width=15, height=2, command=not_improve)
choose2.pack()
windows.mainloop()
# ---------picture loader------------------------------------------------------
#loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
loader = transforms.Compose([transforms.ToTensor()])
# simplest image_loader for just one picture,if want  more need to use the CIFAR10Dataset class


def image_loader(image_name):
    img = Image.open(image_name)
    img = img.resize((imsize, imsize), Image.ANTIALIAS)
    img = loader(img).unsqueeze(0)
    return img.to(device, torch.float)


def imsave(tensor, No):
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = unloader(img)
    img = img.resize(
        (content_origin_width, content_origin_lenth), Image.ANTIALIAS)
    img.save(output_path)
    # img.save("D:\\python_cv\\{}1.jpg".format(No))  #keep the middle result for analysis


'''
filepath = "..."
style_img = image_loader(filepath+'\style.jpg')
content_img = image_loader(file_nema+'\content.jpg')
'''

# change picture size for better transform

style_img = image_loader(style_path)
content_img = image_loader(content_path)
# if style_img.size() != content_img.size() or imsize != style_img.size():
#print("please input style and content of the same size")
# ----------picture print out----------------------------------------------------------------
unloader = transforms.ToPILImage()
plt.ion()


def imshow(tensor, title=None):
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = unloader(img)
    plt.imshow(img)
    if title != None:
        plt.title(title)
    plt.pause(0.01)


if total_turns == 0:
    plt.figure()
    # imshow(style_img,title='Style Image')
    imshow(style_img, title='Style Image')
    plt.pause(1)
    plt.close()
    plt.figure()
    imshow(content_img, title='Content Image')
    plt.pause(1)
    plt.close()
# ---------content loss-------------------------------------------------------------------------------
# cnn kernal part(in class)


class Content_Loss(nn.Module):
    def __init__(self, target,):
        super(Content_Loss, self).__init__()
        self.target = target.detach()

    def forward(self, input):  # forward!
        self.loss = F.mse_loss(input, self.target)
        return input
# define gram_matrix calculate


def gram_matrix(input):
    a, b, c, d = input.size()
    feature = input.view(a*b, c*d)
    gram = torch.mm(feature, feature.t())
    return gram.div(a*b*c*d)
# ----------style loss--------------------------------------------------------------------------------


class Style_Loss(nn.Module):
    def __init__(self, target_featrue):
        super(Style_Loss, self).__init__()
        self.target = gram_matrix(target_featrue).detach()

    def forward(self, input):
        gram = gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input


# -----------load model VGG19---------------------------------------------------------------------------------
'''
cnn = models.vgg19(pretrained=False)
cnn_path = 'D:\\my_cv_style_transfer\\vgg19.pth'
pre = torch.load(cnn_path)
cnn.load_state_dict(pre)
cnn = models.vgg19(pretrained=True)
print(cnn)
'''
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# -------normalize section------------------------------------------------------------------
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        #self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.mean = mean.clone().detach().view(-1, 1, 1)
        #self.std = torch.tensor(std).view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img-self.mean)/self.std


# -----------insert loss function to model----------------------------------------------
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_loss(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(
        normalization_mean, normalization_std).to(device)
    content_loss_amount = []
    style_loss_amount = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i = i+1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            #error report
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target_feature = model(content_img).detach()
            content_loss = Content_Loss(target_feature)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_loss_amount.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = Style_Loss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_loss_amount.append(style_loss)

    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], Content_Loss) or isinstance(model[i], Style_Loss):
            break
    model = model[:(i+1)]

    return model, style_loss_amount, content_loss_amount


# ------------input sample---------------------------------------------------------------
# usually use original picture as input
# input_img=image_loader("...")#restart from middle result
input_img = image_loader(output_path)
plt.ion()
plt.figure()
imshow(input_img, title='input_image')
plt.pause(1)
plt.close()
# -----------optimize the input-------------------------------------------------------------------------


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# ------------kernal training---------------------------------------------------------------


def run_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1):
    print('structing the style transfer model......')
    model, style_loss_amount, content_loss_amount = get_style_model_and_loss(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)
    print('optimizing......')

    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = 0
        content_score = 0

        for s_loss_list in style_loss_amount:
            style_score += s_loss_list.loss
        for c_loss_list in content_loss_amount:
            content_score += c_loss_list.loss

        style_score *= style_weight
        content_score *= content_weight

        total_loss = style_score+content_score
        total_loss.backward()

        run[0] += 1

        print("ren{}:".format(run))
        print('Style_Loss:{:4f} Content_Loss:{:4f}\n'.format(
            style_score.item(), content_score.item()))
        if run[0] % 20 == 0:
            plt.ion()
            plt.figure()
            imshow(input_img)
            #imsave(input_img, (run[0]-(run[0] % 50))/50)
            plt.pause(2)
            plt.close()
        return style_score+content_score
    run = [0]  # time,tracer

    while run[0] <= num_steps:
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

# --------------set content and style weight------------------------------------------------


while True:#a recycle for several solution,which can restart from middle result
    

    content_weight = 0  # change the weight of style content
    windows = tk.Tk()
    windows.title("set wanted weight of content image")
    windows.geometry('340x600')
    image = Image.open("D:\python\pythontest\photos\sailboat_transformed.jpg")
    photo = ImageTk.PhotoImage(image)
    theLabel = tk.Label(windows, width=250, height=250, text="choose your wish", justify=tk.LEFT,
                        image=photo, compound=tk.CENTER, font=("华文行楷", 20), fg="black")  #
    #theLabel.grid(row=1, column=4)
    theLabel.pack()
    l = tk.Label(windows, bg='blue', fg='white', width=20, text='empty')
    #l.grid(row=5, column=3)
    l.pack()

    def print_selection(v):
        l.config(text='you have selected ' + v)
        global content_weight
        content_weight = s.get()

    def close():
        print(content_weight)
        windows.destroy()

    def break_out():
        global cmd_end
        cmd_end = 1
        windows.destroy()

    s = tk.Scale(windows, label='set favored weight', from_=0, to=1, orient=tk.HORIZONTAL, length=300,
                 showvalue=0, tickinterval=2, resolution=0.01, command=print_selection)
    B = tk.Button(windows, text="confirmed", bg="cyan",
                  width=15, height=2, command=close)

    cmd2 = tk.Button(windows, text="end program",  bg="blue",
                     width=15, height=2, command=break_out)
    #s.grid(row=6, column=4)
    s.pack()
    #cmd2.grid(row=8, column=3)
    cmd2.pack()
    #B.grid(row=10, column=3)
    B.pack()
    windows.mainloop()
    # read in value as parameter for transfer
    

    # ------------begin training!!  ohhhhhhh----------------------------------------------------
    if cmd_end == 1:
        break
    output = run_style_transfer(cnn, cnn_normalization_mean,
                                cnn_normalization_std, content_img, style_img, input_img, 300, 1000000, content_weight)
    '''
    if flag==True:
        ROI_input=run_style_transfer(cnn, cnn_normalization_mean,
                                cnn_normalization_std, ROI_content, ROI_style, ROI, 300, 1000000, content_weight)
    
        img = cv2.erode(ROI_input, rgb)  # corrode
        img = cv2.dilate(img, rgb)  # swolen
        mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #get the mask and shape
        ROI = cv2.bitwise_not(mask, output)
        output=cv2.bitwise_and(output,ROI)
        '''
    plt.figure()
    imshow(output, title='output_transformed_image')
    imsave(output, 100)
    plt.ioff()
    plt.show()
    plt.pause(2)
    plt.close()
    total_turns += 1



    #-------------end------------------------------------supported by:Yusen Wu------------------
