# For SVM, we have considered 3D fMRI data from ABIDE
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import nibabel as nib #reading MR images
from sklearn.model_selection import train_test_split
import math
import glob
from matplotlib import pyplot as plt
import os

file_path = glob.glob('H:/ABIDE_Initiative/ABIDE/func_mean/dparsf_filt_global/NYU/nyu_train/tc/*')
#save_path = 'H:/ABIDE_Initiative/ABIDE/func_mean/ccs_filt_global/caltech/train/asd/'

img_train_asd = []
for f in range(len(file_path)):
    a = nib.load(file_path[f])
    a = a.get_data()
    a = a[:,23:73,:]
    for i in range(a.shape[1]):
        img_train_asd.append((a[:,i,:]))
print(a.shape)
print(a[:,0,:].shape)

img_train_asd = np.asarray(img_train_asd)
print(img_train_asd.shape)

img_train_asd = img_train_asd.reshape(-1, 61, 61, 1)
print(img_train_asd.shape)

m = np.max(img_train_asd)
mi = np.min(img_train_asd)
img_train_asd = (img_train_asd - mi)/(m - mi)

print(np.max(img_train_asd), np.min(img_train_asd))

temp = np.zeros([5250, 64, 64, 1])
temp[:,3:,3:,:] = img_train_asd
img_train_asd = temp

print(img_train_asd.shape)

#original_dir = 'D:/ABIDE_Initiative/caltech_train'
#my_path = os.path.join(original_dir, 'asd22')
#os.mkdir(my_path)

#plt.figure()
#plt.subplot(111)

#for i in range(len(img_train_asd)):
  #  img_train_asd = np.reshape(img_train_asd[i], (62, 62))
  #  plt.imshow(img_train_asd, cmap='gray')
    #plt.imsave('asd1990.jpg', i)
   # plt.savefig(os.path.join(my_path, 'asd%d' %i + ".jpg"))
    #plt.show()

from sklearn.model_selection import train_test_split
train_x, valid_x, train_ground, valid_ground = train_test_split(img_train_asd,
                                                                img_train_asd,
                                                                test_size=0,
                                                                random_state=13)

print("Dataset (images), shape: {shape}".format(shape=img_train_asd.shape))
print(len(train_x))
#print(len(valid_x)) 130
plt.figure(figsize=[5, 5])
#Display the images and save it
plt.subplot(111)
for i in range(5250):
    img = np.reshape(train_x[i], (64, 64))
    #plt.imshow(img, cmap='gray')
    plt.imsave('H:/ABIDE_Initiative/ABIDE/func_mean/dparsf_filt_global/NYU/Train/tc_1/' + str(i) , img)
#plt.imshow(img, cmap="gray")
#plt.show()
