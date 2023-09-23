extern "C" {

  int const threadsPerBlock = sizeof(unsigned long long) * 8;
  __device__ inline int ceil_div(int a, int b){
    return (int) (a + b - 1) / b;
  }

  __device__ inline bool devIoU(
      float const* const a,
      float const* const b,
      const float threshold) {
    float left = max(a[0], b[0]), right = min(a[2], b[2]);
    float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
    float width = max(right - left, (float)0), height = max(bottom - top, (float)0);
    float interS = (float)width * height;
    float Sa = ((float)a[2] - a[0]) * (a[3] - a[1]);
    float Sb = ((float)b[2] - b[0]) * (b[3] - b[1]);
    return (interS / (Sa + Sb - interS)) > threshold;
  }

  __global__ void nms_kernel_impl(
      int n_boxes,
      double iou_threshold,
      const float* dev_boxes,
      unsigned long long* dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    if (row_start > col_start)
      return;

    const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 4];
    if (threadIdx.x < col_size) {
      block_boxes[threadIdx.x * 4 + 0] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
      block_boxes[threadIdx.x * 4 + 1] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
      block_boxes[threadIdx.x * 4 + 2] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
      block_boxes[threadIdx.x * 4 + 3] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
      const float* cur_box = dev_boxes + cur_box_idx * 4;
      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = threadIdx.x + 1;
      }
      for (i = start; i < col_size; i++) {
        if (devIoU(cur_box, block_boxes + i * 4, iou_threshold)) {
          t |= 1ULL << i;
        }
      }
      const int col_blocks = ceil_div(n_boxes, threadsPerBlock);
      dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
  }
}