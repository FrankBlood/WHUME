from scanner import SliderWindow
import os
import cv2

def makeimage(cv2_image, stepSizeX, stepSizeY, windowSizeW, windowSizeH):
    allImage = SliderWindow.run(cv2_image, stepSizeX, stepSizeY, windowSizeW, windowSizeH)
    i = 0
    for img in allImage:
        cv2.imwrite('E:/Python/WHUME/createdImage/'+ str(i) +'.jpg', img[2])
        i = i + 1
        print 'Successful!'

def findimage(dirname):
    filelist = os.listdir(dirname)
    return filelist

if __name__ == '__main__':
    dirname = './dataset/car'
    for file in findimage(dirname):
        imgdirname = dirname+'/'+file
        img = cv2.imread(imgdirname)
        makeimage(img, 32, 32, 64, 64)