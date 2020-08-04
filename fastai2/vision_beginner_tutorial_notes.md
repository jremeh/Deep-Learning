# Fast.ai 2 Vision Tutorial (Beginners)


- Single-label classification
- Multiclass classification
- Multilabel classification
- Segment classification
- Points prediction

### Single-label classification
Get data ready for model by putting it in a DataLoaders object
To use it with a function that labels file names, we will use `ImageDataLoaders.from_name_func`

```
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
```

`item_tfms` is a transformation we apply on all our items in our fataset that will resize each image to 224 by 224

create a ```Learner```, a fastai object that combines the data and a model for training and uses transfer learning to fine tune a pretrained model in just two lines of code.

Since it's pretty common to use regular expressions to label the data (often, labels are hidden in the file names), there is a factory method to do just that:
```ImageDataLoaders.from_name_re```

Key takeaway
```
dls =
ImageDataLoaders.from_name_func
ImageDataLoaders.from_name_re
```

### Breed Classification (Multiclass)
Since classifying the exact breed of cats or dogs amongst 37 different breeds is a harder problem, we will slightly change the definition of our DataLoaders to use data augmentation:
```
dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(460),
                                    batch_tfms=aug_transforms(size=224))
```

 We resized to a larger size before batching, and using ```batch_tfms=aug_tranforms```, provides a collection of data augmentation transform with defaults taht are found to worked very well on most datasets

 ```
 learn = cnn_learner(dls, resnet34, metrics=error_rate)
 ```
 
 Another thing that is useful is an interpretation object, it can show us where the model made the worse predictions:
```
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))
```

## Multilabel Classification
```
dls = ImageDataLoaders.from_df(df, path, folder='train', valid_col='is_valid', label_delim=' ',
                               item_tfms=Resize(460), batch_tfms=aug_transforms(size=224))
```

with Data Block API
```
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter('is_valid'),
                   get_x=ColReader('fname', pref=str(path/'train') + os.path.sep),
                   get_y=ColReader('labels', label_delim=' '),
                   item_tfms = Resize(460),
                   batch_tfms=aug_transforms(size=224))
```

## Segmentation
```
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter('is_valid'),
                   get_x=ColReader('fname', pref=str(path/'train') + os.path.sep),
                   get_y=ColReader('labels', label_delim=' '),
                   item_tfms = Resize(460),
                   batch_tfms=aug_transforms(size=224))
```
CNN won't work for segmentation hence we'll use UNet
```
learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
```
with data block
```
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items = get_image_files,
                   get_y = label_func,
                   splitter=RandomSplitter(),
                   batch_tfms=aug_transforms(size=(120,160)))
```

## Points
The major point is to make understand the data format and put it into a Data Block, then the rest of the learning is more or less similar to the above methods.