#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// External function from solution.cu
extern "C" void solve(const float* A, const float* B, float* C, int N);

int main() {
    // Input vectors
    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
    int N = A.size();

    // Output vector
    std::vector<float> C(N);

    // Device pointers
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the vector addition function
    solve(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(C.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Vector A: [";
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << (i < N-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "Vector B: [";
    for (int i = 0; i < N; i++) {
        std::cout << B[i] << (i < N-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "Result C: [";
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << (i < N-1 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}