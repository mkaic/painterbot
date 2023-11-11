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

## Environment
I've been developing from inside the [latest PyTorch Docker container from Nvidia (23.10-py3 at the time of writing)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) using VSCode.

```
docker run \
-it \
-d \
--gpus all \
-v /workspace:/workspace \
nvcr.io/nvidia/pytorch:23.10-py3
```

## Goals
My long-term aim with this project is to get the time required to reconstruct an image with 1000 strokes below 10 seconds. If I can achieve this, it would make it feasible for me to mass-encode large datasets of images like Conceptual 12M or LAION-Art. Then, I plan to train a text-conditional autoregressive sequence model on the encoded images. A true PainterBot!

Currently, my code is capable of optimizing a 1024-stroke reconstruction in around 2 minutes.