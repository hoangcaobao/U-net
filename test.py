from model_structure import *
import numpy as np
import cv2
import os 
import glob 
def test(test_folder, save_path):
    model=unet('unet_weights.hdf5')
    img_list=sorted(glob.glob('{}/*.png'.format(test_folder)))
    for i in img_list:
        file_name=os.path.basename(i)
        img=cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(save_path,file_name), img)
        save_size=img.shape
        img=cv2.resize(img, (256,256), interpolation = cv2.INTER_NEAREST_EXACT)
        img=img/255
        img=np.array(img)
        img=np.expand_dims(img, axis=0)
        img=np.expand_dims(img, axis=3)
        predict=model.predict(img)
        predict[predict<=0.5]=0
        predict[predict>0.5]=1
        predict=predict.reshape((256,256))
        img=img[0].reshape((256,256))
        for i in range(256):
            for j in range(256):
                if (predict[i][j]==0):
                    img[i][j]=0
        img=cv2.resize(img, (save_size[1], save_size[0]), interpolation = cv2.INTER_NEAREST_EXACT))
        cv2.imwrite(os.path.join(save_path,file_name), img*255)

test("data/test","result")
        
        
