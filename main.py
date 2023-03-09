import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as im
from PIL import ImageEnhance
from scipy import fftpack as ff
#load input image
impath = "MESSI_image.jpeg"
img = im.open(impath)

#extract data into workable format
img_data = np.array(img)

# create black & white image
img_data_bw = img_data[:,:,0]
img_bw = im.fromarray(np.uint8(img_data))
# plt.imshow(bw_data,cmap='Greys_r')
# plt.show()

#kernel parameter
s = 32
p_range = int(len(img_data)/s)

k_sparse = np.zeros(np.shape(img_data_bw))
unit_base = np.eye(s,s)
for i in range(p_range):
    for j in range(p_range):
        kernel = img_data_bw[s*i:s*(i+1),s*j:s*(j+1)]
        dct_base = ff.dct(kernel)