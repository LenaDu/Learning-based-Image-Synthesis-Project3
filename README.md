# Project3 When Cats meet GANs


#### Lena Du



### Part 1 DCGAN



#### 1.1 Implement Data Augmentation

Deluxe data augmentation helps the model to be more robust.



#### 1.2 DCGAN - Discriminator

**Padding**

The calculation of padding is:
$$
m = \frac{\lfloor n + 2p-K\rfloor}{S}
$$
, where `m` is the output size and `n` is the input size. `p` is the padding, `K` is the kernel size, and `S` is the stride. Given the size is downsampled by scale `2`, we know `n = 2m`. With `K = 4` and `S = 2`, we will have 
$$
2m = \lfloor 2m +2p-4\rfloor
$$


Then,
$$
p = 1
$$
which means padding is 1.



#### 1.3 DCGAN - Generator

The design of the first layer in `DCGenerator` is using `conv`,  instead of `up_conv`.  The idea is to use padding `3`, kernel size `4,` and stride `1` to obtain a `4x4` output. I also replaced `nn.ReLU` with `nn.LeakyReLU` for its better performance.



#### 1.4 Result

As we can see, the result of the `Deluxe` data augmentation + full `diffaug` configuration with more iterations has better quality and resolution.

| Config                                        | Real Image                                                   | 1000 Iterations                                              | 7000 Iterations                                              |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Basic                                         | ![basic_real](img\DCGAN\basic_real.png)                      | ![basic_real](img\DCGAN\basic_1000.png)                      | ![basic_real](img\DCGAN\basic_7000.png)                      |
| Deluxe                                        | ![basic_real](img\DCGAN\deluxe_real.png)                     | ![basic_real](img\DCGAN\deluxe_1000.png)                     | ![basic_real](img\DCGAN\deluxe_7000.png)                     |
| Deluxe + diffaug (cutout)                     | ![basic_real](img\DCGAN\deluxe_diffaut(cutout)_real.png)     | ![basic_real](img\DCGAN\deluxe_diffaut(cutout)_1000.png)     | ![basic_real](img\DCGAN\deluxe_diffaut(cutout)_7000.png)     |
| Deluxe + diffaug (color, translation)         | ![basic_real](img\DCGAN\deluxe_diffaut(color,tranlation)_real.png) | ![basic_real](img\DCGAN\deluxe_diffaut(color,tranlation)_1000.png) | ![basic_real](img\DCGAN\deluxe_diffaut(color,tranlation)_7000.png) |
| Deluxe + diffaug (color, cutout)              | ![basic_real](img\DCGAN\deluxe_diffaut(color,cutout)_real.png) | ![basic_real](img\DCGAN\deluxe_diffaut(color,cutout)_1000.png) | ![basic_real](img\DCGAN\deluxe_diffaut(color,cutout)_7000.png) |
| Deluxe + diffaug (color, translation, cutout) | ![basic_real](img\DCGAN\deluxe_diffaug_real.png)             | ![basic_real](img\DCGAN\deluxe_diffaug_1000.png)             | ![basic_real](img\DCGAN\deluxe_diffaug_7000.png)             |



| Config                 | Loss                                                   |
| ---------------------- | ------------------------------------------------------ |
| Basic                  | ![basic](img\tensorboard\dcgan\basic.png)              |
| Deluxe                 | ![basic](img\tensorboard\dcgan\deluxe.png)             |
| Deluxe + diffaug (all) | ![basic](img\tensorboard\dcgan\deluxe_all_diffaug.png) |





### Part 2 CycleGAN

*Observations:*

- Generally, with more iteration, the result is better.
- If the Russian blue cat image has eye color other than those of most of the blue cat images, then the model is hard to generate the eyes of grumpy cat result.
- The quality of results is improved with *Cycle Consistency*.
- *Patch* discriminator has better performance than the *DC* discriminator.



#### 2.1 Cat 

##### 2.1.1 DC Discriminator /wo Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\cat_dc_naive\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_naive\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_naive\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\cat_dc_naive\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_naive\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_naive\sample-010000-Y-X.png) |



##### 2.1.2 DC Discriminator /w Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\cat_dc_cycle\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_cycle\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_cycle\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\cat_dc_cycle\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_cycle\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_dc_cycle\sample-010000-Y-X.png) |



##### 2.1.3 Patch Discriminator /wo Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\cat_patch_naive\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_naive\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_naive\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\cat_patch_naive\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_naive\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_naive\sample-010000-Y-X.png) |





##### 2.1.4 Patch Discriminator /w Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\cat_patch_cycle\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_cycle\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_cycle\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\cat_patch_cycle\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_cycle\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\cat_patch_cycle\sample-010000-Y-X.png) |



#### 2.2 Fruits

##### 2.2.1 DC Discriminator /wo Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_naive\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_naive\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_naive\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_naive\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_naive\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_naive\sample-010000-Y-X.png) |



##### 2.2.2 DC Discriminator /w Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_cycle\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_cycle\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_cycle\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_cycle\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_cycle\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_dc_cycle\sample-010000-Y-X.png) |



##### 2.2.3 Patch Discriminator /wo Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_naive\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_naive\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_naive\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_naive\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_naive\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_naive\sample-010000-Y-X.png) |





##### 2.2.4 Patch Discriminator /w Cycle Consistency

| Direction | 1000 Iterations                                              | 5000 Iterations                                              | 10000 Iterations                                             |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| X -> Y    | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_cycle\sample-001000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_cycle\sample-005000-X-Y.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_cycle\sample-010000-X-Y.png) |
| Y -> X    | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_cycle\sample-001000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_cycle\sample-005000-Y-X.png) | ![sample-001000-X-Y](img\CycleGAN\fruit_patch_cycle\sample-010000-Y-X.png) |



### 2.3 Loss

| Discriminator               | Smoothness 0.972                                             |
| --------------------------- | ------------------------------------------------------------ |
| DC /wo Cycle Consistency    | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\dc_cyclegan_NOconsist.png) |
| DC /w Cycle Consistency     | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\dc_cyclegan_consist.png) |
| Patch /wo Cycle Consistency | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\patch_cyclegan_NOconsist.png) |
| Patch /w Cycle Consistency  | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\patch_cyclegan_consist.png) |



| Discriminator               | Smoothness 0.999                                             |
| --------------------------- | ------------------------------------------------------------ |
| DC /wo Cycle Consistency    | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\dc_cyclegan_NOconsist_smooth0.999.png) |
| DC /w Cycle Consistency     | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\dc_cyclegan_consist_smooth0.999.png) |
| Patch /wo Cycle Consistency | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\patch_cyclegan_NOconsist_smooth0.999.png) |
| Patch /w Cycle Consistency  | ![dc_cyclegan_NOconsist](img\tensorboard\cyclegan\patch_cyclegan_consist_smooth0.999.png) |

### Part 3 B&W

####  Spectral loss

| Config            | Real Image                                           | 1000 Iterations                                        | 7000 Iterations                                        |
| ----------------- | ---------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| Deluxe + Instance | ![basic_real](img\DCGAN\deluxe_real.png)             | ![basic_real](img\DCGAN\deluxe_1000.png)               | ![basic_real](img\DCGAN\deluxe_7000.png)               |
| Deluxe + Spectral | ![real-001000](img\bw\spectral loss\real-001000.png) | ![real-001000](img\bw\spectral loss\sample-001000.png) | ![real-001000](img\bw\spectral loss\sample-007000.png) |

#### 
