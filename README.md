# Project3 When Cats meet GANs


### Part 1 DCGAN

#### 1.1 Padding

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

#### 1.2 Results
