
__global__ void softmax_kernel(const float *input, float *output, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N)
    return;

  // Each thread needs to compute the sum of exp of all elements
  // First, find the maximum value for numerical stability
  float max_val = input[0];
  for (int i = 1; i < N; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  // Compute sum of exp(x_i - max_val)
  float sum = 0.0f;
  for (int i = 0; i < N; i++) {
    sum += expf(input[i] - max_val);
  }

  // Compute softmax for this element
  output[idx] = expf(input[idx] - max_val) / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
  cudaDeviceSynchronize();
}
