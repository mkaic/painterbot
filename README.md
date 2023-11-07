# PainterBot
![Mona Lisa reconstructed with 500 strokes](assets/lisa.gif)

## Basic idea
This project was inspired by [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but I have not used any of their code. My algorithm departs conceptually from theirs in a few key places:

1. I'm optimizing 2D gaussians, not 3D.
2. I'm evaluating the gaussians on polar coordinates to allow for variable circular curvature.
3. The quantity and ordering of my gaussians is pre-set from the beginning and static throughout optimization.
4. I am alpha-blending gaussians from back-to-front rather than summing from front-to-back.
5. I optimize a group of strokes, freeze them, then add new strokes and repeat.
6. I can get a pretty good reconstruction with just a few hundred strokes instead of a few million (not really a fair comparison since obviouly I'm solving a much simpler problem, but still)

## Repo structure
* `painterbot.py` contains all the logic and rendering code. `train.py` is just a convenience script I use to call that code. 
* `make_timelapses.sh` takes the frames saved during the optimization process and renders two different timelapse videos: one of the finished painting being painted stroke-by-stroke, and one of the strokes coalescing and optimizing themselves during gradient descent.
* `test.py` loads a small, 250-stroke pre-trained Mona Lisa and renders it with both the python code and (soon) the Triton code.

`triton_render_kernel.py` is where I'm slowly working towards implementing a CUDA kernel for my rendering algorithm using OpenAI's Triton language.

## Environment
I've been developing from inside the [latest PyTorch Docker container from Nvidia (23.10-py3 at the time of writing)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) using VSCode.

```
docker run -it -d --gpus all -v /workspace:/workspace nvcr.io/nvidia/pytorch:23.10-py3
```

## Plans
I am currently working on a Triton implementation of the forward and backward pass of my rendering code. My hope is that this implementation will at least match `torch.compile`'s performance, and ideally surpass it. My long-term aim with this project is to get the time required to reconstruct an image with 1000 strokes below 10 seconds. If I can achieve this, it would make it feasible for me to mass-encode large datasets of images like Conceptual 12M or LAION-Art. Then, I plan to train a text-conditional autoregressive sequence model on the encoded images. A true PainterBot!

Currently, my code is capable of optimizing a 1000-stroke reconstruction in around 4 minutes. `torch.compile` *would* bring that down to 2 minutes if it didn't keep recompiling the darn function every five seconds. My naive hope is that my custom Triton kernel can achieve another 2x over `torch.compile`, bringing 1000 strokes down to 1 minute. From there, I have a few other optimizations I think I can make to speed up training, in order of priority:

1. Progressive resolution growing, a la GANs. The first few batches of strokes can be optimized on a much smaller version of the image, since they are meant to capture the low-frequency structure of the image. A 64x64 image is 64 times smaller than a 512x512 image, meaning it should run 64 times faster. I'll do some tests to figure out what the *least* amount of time is I can get away with optmizing at full-resolution.
2. Running in `bfloat16`. In a perfect world with a perfect kernel, this should net me 50% memory reduction and a bit less than 100% speed increase. In my preliminary tests, this hasn't really been the case unfortunately, but I'm sure I'm just doing something wrong.
3. Investigating alternatives to Adam for the optimizer. I have seen firsthand that Adam is rather sensitive to initial learning rate choice, which I'm not a huge fan of.
4. Not wasting flops on near-0-alpha pixels by not evaluating any coordinate further than, like, 2 sigma from the center of a stroke? Feel like this could speed things up considerably, but would come at the cost of a more complex forward kernel and a hard-to-implement backwards kernel.
