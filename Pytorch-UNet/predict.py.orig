import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def find_largest_polytope(v):
    v1 = 0
    v2 = 1
    v3 = 2
    v4 = 3
    largest_area = -1
    for i in range(v.shape[0]-3):
        for j in range(i+1,v.shape[0]-2):
            for k in range(j+1, v.shape[0]-1):
                for l in range(k+1, v.shape[0]):
                    area = 1/2*((v[i,0]*v[j,1]+v[j,0]*v[k,1]+v[k,0]*v[l,1]+v[l,0]*v[i,1])-\
                        (v[j,0]*v[i,1]+v[k,0]*v[j,1]+v[l,0]*v[k,1]+v[i,0]*v[l,1]))
                    if area > largest_area:
                        largest_area = area 
                        v1 = i 
                        v2 = j 
                        v3 = k
                        v4 = l        
    return hull_vertices[[v1, v2, v3, v4],:]

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # print(np.max(mask))
        # print(np.min(mask))
        pixels = np.where(mask>0)
        # print(pixels)
        pixels_array = np.vstack((pixels[1],pixels[0])).T
        # print(pixels_array.shape)
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.imshow(mask)
        plt.figure()
        plt.imshow(img)
        # plt.show()
        
        import scipy.spatial 
        hull = scipy.spatial.ConvexHull(pixels_array)
        # plt.plot(pixels_array[hull.vertices,0],pixels_array[hull.vertices,1],'r*')
        hull_vertices = pixels_array[hull.vertices,:]
        center_point = np.mean(hull_vertices,axis=0)
        # plt.plot(center_point[0], center_point[1], 'b*')

        pts = find_largest_polytope(hull_vertices)
        
        upleft = np.where((pts[:,0]<center_point[0]) & (pts[:,1]<center_point[1]))
        upleft_vertices = pts[upleft[0],:]
        plt.plot(upleft_vertices[:,0],upleft_vertices[:,1],'r*')

        bottomleft = np.where((pts[:,0]<center_point[0]) & (pts[:,1]>center_point[1]))
        bottomleft_vertices = pts[bottomleft[0],:]
        plt.plot(bottomleft_vertices[:,0],bottomleft_vertices[:,1],'g*')

        bottomright = np.where((pts[:,0]>center_point[0]) & (pts[:,1]>center_point[1]))
        bottomright_vertices = pts[bottomright[0],:]
        plt.plot(bottomright_vertices[:,0],bottomright_vertices[:,1],'b*')

        upright = np.where((pts[:,0]>center_point[0]) & (pts[:,1]<center_point[1]))
        upright_vertices = pts[upright[0],:]
        plt.plot(upright_vertices[:,0],upright_vertices[:,1],'y*')


        # plt.plot(pts[:,0], pts[:,1],'g*')
        # bottomleft = np.where((hull_vertices[:,0]<center_point[0]) & (hull_vertices[:,1]>center_point[1]))
        # bottomleft_vertices = hull_vertices[bottomleft[0],:]
        # diff = bottomleft_vertices - center_point
        # tmp = np.argmax(np.linalg.norm(diff,axis=1)*np.cos(np.arctan2(diff[:,1], diff[:,0]))**2*np.sin(np.arctan2(diff[:,1], diff[:,0]))**2)
        # bottomleft_vertex = bottomleft_vertices[tmp,:]
        # # plt.plot(hull_vertices[bottomleft,0], hull_vertices[bottomleft,1],'g*')
        # plt.plot(bottomleft_vertex[0], bottomleft_vertex[1], 'g*')
        
        # upleft = np.where((hull_vertices[:,0]<center_point[0]) & (hull_vertices[:,1]<center_point[1]))
        # upleft_vertices = hull_vertices[upleft[0],:]
        # diff = upleft_vertices - center_point
        # tmp = np.argmax(np.linalg.norm(diff,axis=1)*np.cos(np.arctan2(diff[:,1], diff[:,0]))**2*np.sin(np.arctan2(diff[:,1], diff[:,0]))**2)
        # upleft_vertex = upleft_vertices[tmp,:]
        # # plt.plot(hull_vertices[upleft,0], hull_vertices[upleft,1],'g*')
        # plt.plot(upleft_vertex[0], upleft_vertex[1], 'g*')
        
        # bottomright = np.where((hull_vertices[:,0]>center_point[0]) & (hull_vertices[:,1]>center_point[1]))
        # bottomright_vertices = hull_vertices[bottomright[0],:]
        # diff = bottomright_vertices - center_point
        # tmp = np.argmax(np.linalg.norm(diff,axis=1)*np.cos(np.arctan2(diff[:,1], diff[:,0]))**2*np.sin(np.arctan2(diff[:,1], diff[:,0]))**2)
        # bottomright_vertex = bottomright_vertices[tmp,:]
        # # plt.plot(hull_vertices[bottomright,0], hull_vertices[bottomright,1],'g*')
        # plt.plot(bottomright_vertex[0], bottomright_vertex[1], 'g*')
        
        # upright = np.where((hull_vertices[:,0]>center_point[0]) & (hull_vertices[:,1]<center_point[1]))
        # upright_vertices = hull_vertices[upright[0],:]
        # diff = upright_vertices - center_point
        # tmp = np.argmax(np.linalg.norm(diff,axis=1)*np.cos(np.arctan2(diff[:,1], diff[:,0]))**2*np.sin(np.arctan2(diff[:,1], diff[:,0]))**2)
        # upright_vertex = upright_vertices[tmp,:]
        # # plt.plot(hull_vertices[upright,0], hull_vertices[upright,1],'g*')
        # plt.plot(upright_vertex[0], upright_vertex[1], 'g*')
        
        plt.show()

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask, mask_values)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
