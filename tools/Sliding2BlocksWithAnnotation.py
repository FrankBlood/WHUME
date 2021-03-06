# -*- coding: utf-8 -*-
import argparse, os
import cv2
from scanner import SliderWindow 
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-WW", help="the weight of slider window", type=int)
    parser.add_argument("-WH", help="the height of slider window", type=int)
    parser.add_argument("-SX", help="stride of x", type=int)
    parser.add_argument("-SY", help="stride of y", type=int)
    parser.add_argument("-I", help="input path, list all image file path per line or folder path", type=str)
    parser.add_argument("-O", help="output folder path, the blocks of images will be saved to", type=str)
    args = parser.parse_args()
    
    # 设置slider window
    slider_w = SliderWindow()
    
    assert args.I != None
    assert args.O != None
    assert args.WW != None
    assert args.WH != None
    assert args.SX != None
    assert args.SY != None

    if not os.path.exists(args.O):
        os.mkdir(args.O)
    if not os.path.exists(args.O+"/0/"):
        os.mkdir(args.O+"/0/")
    if not os.path.exists(args.O+"/1/"):
        os.mkdir(args.O+"/1/")

    if os.path.isfile(args.I):
        fp = open(args.I, 'r')
        for line in fp.readlines():
            img = cv2.imread(line.strip())
            img_name = line[line.rfind("/")+1:].strip()
            n = 1
            for x, y, block_img in slider_w.run(img, args.SX, args.SY, args.WW, args.WH):
                h, w, c = np.shape(block_img)
                if w == args.WW and h == args.WH:
                    copy_img = img.copy()
                    cv2.rectangle(copy_img, (x, y), (x + args.WW, y + args.WH), (0, 255, 0), 2)
                    cv2.imshow("Window", copy_img)
                    key = cv2.waitKey(0)
                    if key == 27: # ESC
                        cv2.imwrite(args.O+"/0/"+str(n)+"-"+img_name, block_img) 
                        n += 1
                    elif key == 32: # space
                        cv2.imwrite(args.O+"/1/"+str(n)+"-"+img_name, block_img) 
                        n += 1
        fp.close()
    elif os.path.isdir(args.I):
        for img_name in os.listdir(args.I):
            if img_name.endswith(".jpg"):
                img_path = args.I+"/"+img_name
                img = cv2.imread(img_path)
                n = 1
                for x, y, block_img in slider_w.run(img, args.SX, args.SY, args.WW, args.WH):
                    h, w, c = np.shape(block_img)
                    if w == args.WW and h == args.WH:
                        copy_img = img.copy()
                        cv2.rectangle(copy_img, (x, y), (x + args.WW, y + args.WH), (0, 255, 0), 2)
                        cv2.imshow("Window", copy_img)
                        key = cv2.waitKey(0)
                        if key == 27: # ESC
                            cv2.imwrite(args.O+"/0/"+str(n)+"-"+img_name, block_img) 
                            n += 1
                        elif key == 32: # space
                            cv2.imwrite(args.O+"/1/"+str(n)+"-"+img_name, block_img) 
                            n += 1
