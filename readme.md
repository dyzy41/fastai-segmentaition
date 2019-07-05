这个代码主要数数据的摆放一定要正确，codes.txt里面放的图片的总类别。valid.txt似乎放的是验证集图片的名字。img里面是切割好的图片，gt里面是0-1图，如果是多类别，就是0-15的灰度图。
训练出来的model会保存在img/models。你们把我们的数据按照规则摆放好，就能跑了。
env：python3.6， pytorch1.0+, fastai1.0+
可以全部安装最新版的。
代码里面的路径按规则设置好。
代码不懂的查doc，或者群里问我，训练前要下载resnet的模型，直接打开链接下载，下完以后，放在/home/kawhi/.cache/torch/checkpoints/*

