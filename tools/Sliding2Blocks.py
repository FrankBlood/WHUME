# -*- coding: utf-8 -*-

import argparse, os
import cv2
from scanner import SliderWindow 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-WW", help="the weight of slider window", type=int)
    parser.add_argument("-WH", help="the height of slider window", type=int)
    parser.add_argument("-SX", help="stride of x", type=int)
    parser.add_argument("-SY", help="stride of y", type=int)
    parser.add_argument("-I", help="input file path, list all image file path per line", type=str)
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

    fp = open(args.I, 'r')
    for line in fp.readlines():
        img = cv2.imread(line.strip())
        img_name = line[line.rfind("/")+1:].strip()
        n = 1
        for x, y, block_img in slider_w.run(img, args.SX, args.SY, args.WW, args.WH):
            cv2.imwrite(args.O+"/"+str(n)+"-"+img_name, block_img) 
            n += 1
    fp.close()
