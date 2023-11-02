# PainterBot
![Mona Lisa reconstructed with 500 strokes](assets/lisa.gif)

## Basic idea
This project was inspired by [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but I have not used any of their code. My algorithm departs conceptually from theres in a few key places:

1. I'm optimizing 2D gaussians, not 3D.
2. I'm evaluating the gaussians on polar coordinates to allow for variable circular curvature.
3. The quantity and ordering of my gaussians is pre-set from the beginning and static throughout optimization.
4. I am alpha-blending gaussians from back-to-front rather than summing from front-to-back.
5. I optimize a group of strokes, freeze them, then add new strokes and repeat.
6. I can get a pretty good reconstruction with just a few hundred strokes instead of a few million (not really a fair comparison since obviouly I'm solving a much simpler problem, but still)

## Repo structure
`painterbot.py` contains all the logic and rendering code. `invoke.py` is just a convenience script I use to call that code. `make_timelapses.sh` takes the frames saved during the optimization process and renders two different timelapse videos: one of the finished painting being painted stroke-by-stroke, and one of the strokes coalescing and optimizing themselves during gradient descent.

`render_forward.cu` and `render_backward.cu` are both empty for the moment. CUDA is terrifying.

## Environment
I've been developing from inside the [latest PyTorch Docker container from Nvidia (23.10-py3 at the time of writing)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags) using VSCode.

## Devlog
`2023 - 11 - 02`
I believe I've gotten this code running as quickly as I can realistically expect on my 3090 using pure PyTorch. If I `torch.compile` my `Renderer` class, it can rasterize and layer about 3000 gaussians per second, forwards *and* backwards. So computationally it's more like 6000 gaussians per second. I have found that 1000 total strokes, each optimized for 300 iterations, yields a pretty good baseline reconstruction. That's 600,000 total gaussian evaluations required if you count both forward and backward passes. If I could get torch to stop recompiling my Renderer for each new group of strokes, this would theoretically take around 2-3 minutes. I strongly *believe* it should be possible to go *much* faster than this.

To me, the next logical step is to implement my core rendering code as a custom CUDA kernel. My rendering algorithm is pixel-independent, meaning it *should* parallelize really well. I have never implemented a custom CUDA kernel before, though, so this is going to be an adventure. In the best case, this custom implementation will net me 100x improvements and I'll be able to reconstruct images in seconds rather than minutes. In the *realistic* case, I might make it like 2 times faster. Oh well—only one way to know for sure!
