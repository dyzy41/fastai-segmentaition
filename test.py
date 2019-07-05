from fastai.vision import *
from fastai.callbacks.hooks import *
import matplotlib.pyplot as plt
import cv2

import warnings
warnings.filterwarnings("ignore")
path = './small_data'

path_lbl = path+'/'+'gt'
path_img = path+'/'+'img'
fnames = get_image_files(path_img)
lbl_names = get_image_files(path_lbl)

img_f = fnames[0]
img = open_image(img_f)
# img.show(figsize=(5,5))

get_y_fn = lambda x: path_lbl+'/'+f'{x.stem}_P{x.suffix}'
mask = open_mask(get_y_fn(img_f))
# mask.show(figsize=(5,5), alpha=1)
src_size = np.array(mask.shape[1:])
codes = np.loadtxt(path+'/'+'codes.txt', dtype=str)
size = src_size
bs=2
src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('/media/kawhi/08DB0A6C08DB0A6C/1Ubuntu_extend/datset/adv_samples/fastai_seg/small_data/valid.txt')   #juedui lujing!!!
       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs, num_workers=4)
        .normalize(imagenet_stats))

name2id = {v:k for k,v in enumerate(codes)}
void_code = 24

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

metrics=acc_camvid
wd=1e-2
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
# learn.fit_one_cycle(20)
learn.load('model')

img2 = open_image('./small_data/img/austin1_4.jpg')
# img2.resize(224)
# img2.resize(size=size)
# img_pred = learn.predict(img2)
# img2.show(y=learn.predict(img2)[0],figsize=(5,5))



# plt.imshow(img_pred[0].data)
temp = np.array(learn.predict(img2)[0].data)

temp = temp.squeeze()
cv2.imwrite('0.jpg', temp)
# plt.imshow(temp)
# plt.savefig("0.png")

# lr_find(learn)
# learn.recorder.plot()

