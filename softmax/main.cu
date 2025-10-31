#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Declare the solve function from solution.cu
extern "C" void solve(const float *input, float *output, int N);

int main() {
    // Example input size
    const int N = 3;

    // Create input data on host
    std::vector<float> h_input = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_output(N);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the solve function
    std::cout << "Calling solve function..." << std::endl;
    solve(d_input, d_output, N);

    // Copy result back from device to host
    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Input:  ";
    for (int i = 0; i < N; i++) {
        std::cout << h_input[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "Done!" << std::endl;
    return 0;
}