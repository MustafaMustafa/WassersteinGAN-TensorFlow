## WassersteinGAN

TensorFlow implementation of [Wasserstein GAN](https://arxiv.org/abs/1701.07875)  

Other implementations:
- [Torch (Author's)](https://github.com/martinarjovsky/WassersteinGAN)  
- [Chainer](https://github.com/hvy/chainer-wasserstein-gan)  
- [Keras](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/WassersteinGAN)  
  

DCGAN model/ops are a modified version of Taehoon Kim's implementation [@carpedm20](https://github.com/carpedm20/DCGAN-tensorflow).

- - -
### Usage

Download dataset:
```bash
python download.py celebA
```
To train:
```bash
python main.py --dataset celebA --is_train --is_crop
```

*Note:* a NumPy array of the input data is created by default. This is to avoid batch by batch IO. 
You can turn this option off if the available memory is too small on your system or if your dataset is too large.
```bash
python main.py --dataset celebA --is_train --is_crop --preload_data False
```
