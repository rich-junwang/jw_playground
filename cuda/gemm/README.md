[//]: # (https://siboehm.com/articles/22/CUDA-MMM)

1. Full correctness + performance test, only naive

```bash
DEVICE=0 ./gemm 1
```


2. Launch naive once (no benchmark loop) -- pick M, K, N
```bash
DEVICE=0 ./gemm --once 1 512  512 512
```


3. Confirm indices

```bash
./gemm --list-kernels
```