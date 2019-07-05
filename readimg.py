from PIL import Image
import numpy as np
import cv2
p = '/media/kawhi/08DB0A6C08DB0A6C/1Ubuntu_extend/datset/adv_samples/0.jpg'
img = Image.open(p)
x = np.asarray(img)
x = np.where(x==0, 0, 255)
cv2.imwrite('1.jpg', x.astype(np.uint8))
print(set(x.flatten()))
print('ok')