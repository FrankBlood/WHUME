from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from keras.utils import np_utils
import cv2
import re

def makeImage(file_path):
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode = 'nearest')
    
    with open(file_path, 'r') as fp:
        print 'open correctly!'
        for line in fp.readlines():
            img_path = line.strip().split()[0]
            img = load_img(img_path)
            img_name = re.split(r'[\/\.]+', img_path)
            x = img_to_array(img)
            print 'load image correctly!'
            x = x.reshape((1,)+x.shape)
            y = np.array([[0]],dtype=np.uint8)
            y = np_utils.to_categorical(y, 1)
            i = 0
            for x_batch, y_batch in datagen.flow(x, y, batch_size=1):
                img = array_to_img(x_batch[0])
                img.save("./preview/"+img_name[2]+'_'+str(i)+".jpg")
                print 'save image correctly'
                with open('new_train.txt', 'w') as fp:
                    fp.write("./preview/"+img_name[2]+'_'+str(i)+".jpg"+' '+str(line.strip().split()[1])+'\n')
                i += 1
                if i > 20:
                   break

if __name__ == '__main__':
    makeImage('phone_train_set.txt')
