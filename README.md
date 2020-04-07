# mass_road_extraction
Massachusetts Road extraction by u-net using fastaiv1
@[TOC]([fast.ai] unet道路提取)
前一篇介绍了[unet网络对Camvid数据集的分](https://blog.csdn.net/qq_39337332/article/details/105335457)类，道路提取的问题可以看成简化的Segmentation：就是将像素点转化为两类`['0', '1']`。

`Massachusetts roads`数据集是Volodymyr Mnih的[博士项目](https://www.cs.toronto.edu/~vmnih/data/)，搜集了马萨诸塞州的遥感图片并分别对建筑物和路径进行了标定，这里我们只用到了路径的数据。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407195356331.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70 =300x300)
下载数据的脚本在下面的github链接中。

下载下来的数据有很多坏掉的图片，即遥感图片有大部分缺失，这些图片我在前期手动进行了剔除。然后因为每一幅原始的`tiff`遥感数据过大，不好直接进行训练。因此进行了随机剪切，一幅图片随机切出15张`256x256`大小的对应图片作为训练数据。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407195849271.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
切分好的数据我放在了下面的百度网盘中：

> 链接:https://pan.baidu.com/s/1l5OytHjQzx2VxYOmq_nEYg  密码:k4kk

现在可以用fastai进行训练了，先导入所需的包

```python
from fastai.vision import *
```
设置训练使用的GPU

```python
torch.cuda.set_device(1)
```
教研室的服务器是4块12GiB的GPU，这里我指定一号GPU。多说一句，fastai现在不支持多GPU进行训练。不过fastai官方给了分布训练的脚本，可以参考这里：
```python
https://docs.fast.ai/distributed.html
```
后续我也会试一下。

```python
mass_roads = Path('~/road_extraction/data/mass_roads_crop')
mass_roads.ls()
```
**OUT**
 
```python
PosixPath('~/road_extraction/data/mass_roads_crop/train'),
PosixPath('~/road_extraction/data/mass_roads_crop/valid')]
```
虽然Volodymyr Mnih给我们划分好了验证集，但是数据数量很少，不太好进行训练，因此最后还是采取了从训练集中划分出20%作为验证集的方式。

```python
mass_roads_train = mass_roads / 'train'
mass_roads_train.ls()
```
**OUT**

```python
 PosixPath('~/road_extraction/data/mass_roads_crop/train/map'),
 PosixPath('~/road_extraction/data/mass_roads_crop/train/sat')]
```
训练集文件中主要就是`sat`和`map`文件夹，`sat`文件夹内存放的就是遥感数据，`map`文件夹内存放的是对应的mask文件。查看一下内容：

```python
sorted(mass_roads_valid.ls()[1].ls())[:5]
```
**OUT**

```python
[PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/map/0_10228690_15.tif'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/map/0_10978735_15.tif'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/map/0_10978795_15.tif'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/map/0_18028945_15.tif'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/map/0_21929020_15.tif')]
```

```python
sorted(mass_roads_valid.ls()[2].ls())[:5]
```
**OUT**

```python
[PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/sat/0_10228690_15.tiff'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/sat/0_10978735_15.tiff'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/sat/0_10978795_15.tiff'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/sat/0_18028945_15.tiff'),
 PosixPath('/home/bir2160400081/road_extraction/data/mass_roads_crop/valid/sat/0_21929020_15.tiff')]
```
可以看到遥感图和mask文件对应的只有后缀名的不同，因此获得标签的函数可以这么写：

```python
get_y_fnc = lambda x: x.parent.parent / 'map' / f'{x.stem}.tif'
```
查看一下某一对数据：

```python
img_sat = sorted(train_sat.ls())[17]
img_map = get_y_fnc(img_sat)

img_sat = open_image(img_sat)
img_map = open_image(img_map)

_, axs = plt.subplots(1, 2, figsize=(5, 5))
img_sat.show(ax=axs[0])
img_map.show(ax=axs[1])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407202805663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70 =300x140)
接下来创建fastai的训练数据对象，首先定义`SegmentationItemList`对象。这个实质上和pytorch的dataset类似，定义的是可取、有长度的对象。不这么做的话，fastai会爆奇怪的mask错误。

```python
class MySegmentationLabelList(SegmentationLabelList):
    def open(self, fn):
        return open_mask(fn, div=True)
    
class MySegmentationItemList(SegmentationItemList):
    _label_cls, _square_show_res = MySegmentationLabelList, False
```
定义训练和验证数据

```python
src = (MySegmentationItemList.from_folder(mass_roads_train / 'sat')
       # Load in x data from folder
#        .split_by_folder(train='train', valid='valid')
       .split_by_rand_pct()
       # Split data into training and validation set 
       .label_from_func(get_y_fnc, classes=['0', '1'])
       # Get label image of sat
)
```
转化为`databunch`对象

```python
tfms = get_transforms()
bs = 32
size= 256
data = (src.transform(tfms, size=size, tfm_y=True)
        # Flip images horizontally 
        .databunch(bs=bs, path=mass_roads_train)
        # Create a databunch
        .normalize(imagenet_stats)
        # Normalize for resnet
)
```
看一下数据什么样：

```python
data.show_batch(figsize=(5, 5))
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407203309460.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
可以看到就算手动剔除了很多差劲的数据，随机剪切的数据图片中仍然有空白情况，不过这种图片很少，并不是很影响训练。

最后定义一个unet训练器，用resnet18提取出来的特征进行图片分割

```python
learn = unet_learner(data, models.resnet18, metrics=dice, wd=1e-2)
```
`dice`是医学影像分割中比较常用的metric，其公式为：
$$
Dice = \frac{A \cap B}{A \cup B}
$$
照例找一下最有学习率

```python
learn.lr_find()
learn.recorder.plot()
```
**OUT**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407210115908.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
最后选取`lr=1e-5`

```python
lr = 1e-5
learn.fit_one_cycle(4, slice(lr), pct_start=0.8)
```
训练上几轮看看结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200407210224856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5MzM3MzMy,size_16,color_FFFFFF,t_70)
到处模型之后 ，预测一下看看效果如何：

```python
learn.path = Path("/home/bir2160400081/fast.ai/mass_road")
learn.save("mass-road-stage-1")
learn.export()
mask_pred = learn.predict(data.train_ds[300][0])
# image_pred
_, axs = plt.subplots(1, 3, figsize=(5, 5))

Image.show(data.train_ds[300][0], ax=axs[0])
Image.show(data.train_ds[300][1], ax=axs[1])
Image.show(mask_pred[0], ax=axs[2])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020040721032181.png)
可见效果不是很好，可能是因为resnet18太浅了；后续利用上文提到的分布式训练再用resnet34或者resnet50试试看；也可以在寻找一下最佳学习率进行进一步的优化，但是我不太看好这个方式。源代码和数据下载的脚本可以在github上找到：
