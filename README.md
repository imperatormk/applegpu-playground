# AppleGPU Playground

A small web frontend for compiling [Triton](https://github.com/triton-lang/triton)
kernels to Apple GPU (AIR) and running them on real Apple Silicon.

**Live:** https://imperatormk.github.io/applegpu-playground/

Write a `@triton.jit` kernel, hit **Compile & Run**, and see each stage of the
pipeline: Triton IR, Triton GPU IR (with the Apple GPU layouts), the emitted
Metal Shading Language, and the final `.metallib`, plus the numerical result
from running it on the GPU.

This repo is **only the static frontend.** It talks to a compile API running on
a real Apple Silicon machine. The backend (the Triton to AIR compiler and the
GPU runner) is experimental and not part of this repo.

## Status

Early and experimental, so expect rough edges. It's a window into an
out-of-tree Apple GPU backend for Triton, not a finished product.
