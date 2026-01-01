# warp级并行规约图示算法
[点击查看 PDF](并行规约.pdf)
```cpp
static __global__ void reduceMax_(fp32_t *odata, const fp32_t *idata, const int L)
{
    // 共享内存：为每个 warp 存储一个最大值，线程块包含 2048 个线程（64 个 warp）
    __shared__ fp32_t warpMin[64]; // 64 个元素，代表 64 个 warp，每个 warp 一个最大值

    // 获取当前线程的全局索引
    unsigned int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_ID = threadIdx.x / WARP_SIZE_1;  // 计算当前线程所属的 warp
    unsigned int lane_ID = threadIdx.x % WARP_SIZE_1;  // 计算当前线程在 warp 内的索引

    fp32_t x = -1e36;  // 初始化一个非常小的值，确保找到最大值
    fp32_t temp = -1e36;  // 临时变量用于比较

    // 如果当前线程索引有效（小于总长度 L），则加载数据
    if (idx < L)
    {
        x = idata[idx];
    }

    // 每个 warp 内的线程进行规约计算，寻找该 warp 内的最大值
    for (int i = WARP_SIZE_1 / 2; i > 0; i = i / 2)
    {
        temp = __shfl_xor(x, i, WARP_SIZE_1);  // 通过 xor 在 warp 内交换数据
        if (x < temp)
        {
            x = temp;
        }
    }

    // 每个 warp 的第一个线程将最大值存储到共享内存 warpMin 中
    if (lane_ID == 0)
    {
        warpMin[warp_ID] = x;
    }
    __syncthreads();  // 同步线程，确保每个 warp 的最大值已存储

    // 规约所有 warp 的最大值，继续在 warp 之间进行最大值比较
    if (threadIdx.x < 64)  // 每个 warp 内的线程进行下一轮规约
    {
        x = warpMin[threadIdx.x];  // 加载共享内存中的每个 warp 的最大值
        for (int i = 64 / 2; i > 0; i = i / 2)
        {
            temp = __shfl_xor(x, i, 64);  // 在 warp 内交换数据，进行规约
            if (x < temp)
            {
                x = temp;
            }
        }

        // 只有第一个线程（即 threadIdx.x == 0）将最终结果写入输出数组 odata
        if (threadIdx.x == 0)
        {
            odata[blockIdx.x] = x;  // 保存每个块的最大值
        }
    }
}
