import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torchvision.transforms as transforms
from net import Net
import matplotlib.pyplot as plt
import cv2
import time

def draw_box_with_label(image, box, label, color=(0, 255, 0)):
    (w, h) = image.size
    font = ImageFont.truetype("arial.ttf", w/30)
    draw = ImageDraw.Draw(image)
    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline ="red", width=int(w/100))
    draw.text((box[0], box[1]), label, font=font, stroke_width=4, stroke_fill='black')

    image.save("prediction.jpg", "JPEG")

def main(image_file, backbone, weights):
    # load net model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = weights
    model = Net(backbone, 2).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # load and convert image
    mean = [0.47964323, 0.4472308,  0.39698297]
    std = [0.23054191, 0.22776432, 0.22883151]

    size = 224

    # read and convert image
    transform = transforms.Compose([
        transforms.ToPILImage(), # pixel to range [0,1]
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image = Image.open(image_file).convert('RGB')
    o_image = image.copy()
    (w, h) = image.size
    image = image.resize((size, size), Image.BICUBIC)
    image = np.array(image, dtype="float32")
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image)
    image = transform(image).to(device)
    image = image.unsqueeze(0)

    # predict 
    label2str = {0: "cat", 1: "dog"}
    # exec_time = 0
    # for i in range(0,10010):
    #     start_time = time.time()
    #     bbox, label = model(image)
    #     if i>10:
    #         exec_time += time.time() - start_time
    # print(f"{exec_time/10000*1000} ms")
    bbox, label = model(image)
    bbox = bbox.cpu().detach().numpy()[0]
    label = label2str[int(label.cpu().detach().numpy().argmax(1))]
    print(bbox)
    print(label)

    xmin, ymin, xmax, ymax = bbox
    xmin *= w
    xmax *= w
    ymin *= h
    ymax *= h

    draw_box_with_label(o_image, [xmin, ymin, xmax, ymax], label)
    o_image.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_file", help="image file")
    parser.add_argument("backbone", help="backbone")
    parser.add_argument("weights", help="weights")
    args = parser.parse_args()
    main(args.image_file, args.backbone, args.weights)