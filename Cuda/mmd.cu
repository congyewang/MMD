#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel to compute the RBF kernel between two sets of points
__device__ float rbfKernel(const float* x, const float* y, int dim, float sigma) {
    float distanceSquared = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - y[i];
        distanceSquared += diff * diff;
    }
    return exp(-distanceSquared / (2.0f * sigma * sigma));
}

// Function to compute the Maximum Mean Discrepancy (MMD) between two sets of points
__global__ void computeMMD(float* samplesP, float* samplesQ, int m, int n, int dim, float sigma, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m + n) return;
    
    float term1 = 0.0f;
    float term2 = 0.0f;
    float term3 = 0.0f;

    if (i < m) {
        // Compute term1
        for (int j = 0; j < m; ++j) {
            term1 += rbfKernel(samplesP + i * dim, samplesP + j * dim, dim, sigma);
        }
        term1 /= float(m * m);
        
        // Compute term2
        for (int j = 0; j < n; ++j) {
            term2 += rbfKernel(samplesP + i * dim, samplesQ + j * dim, dim, sigma);
        }
        term2 /= float(m * n);
        term2 *= 2.0f;

        // Update result
        atomicAdd(result, term1 - term2);
    } else {
        int idx = i - m;
        // Compute term3
        for (int j = 0; j < n; ++j) {
            term3 += rbfKernel(samplesQ + idx * dim, samplesQ + j * dim, dim, sigma);
        }
        term3 /= float(n * n);
        
        // Update result
        atomicAdd(result, term3);
    }
}

int main() {
    // Example usage
    // float X_h[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    // float Y_h[12] = {7, 6, 5, 4, 3, 2, 1, 1, 8, 0, 2, 5};

    float *X_d, *Y_d, *result_d;
    float result_h = 0.0f;

    // int m = 3;
    // int n = 4;
    // int dim = 3;

    // Example usage
    int m = 1000000;
    int n = 1000000;
    int dim = 2;

    float *X_h = new float[m * dim];
    float *Y_h = new float[n * dim];

    // Generate random numbers for X_h and Y_h
    srand(time(NULL));
    for (int i = 0; i < m * dim; ++i) {
        X_h[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    for (int i = 0; i < n * dim; ++i) {
        Y_h[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    float sigma = 1.0f;

    // Allocate device memory
    cudaMalloc(&X_d, m * dim * sizeof(float));
    cudaMalloc(&Y_d, n * dim * sizeof(float));
    cudaMalloc(&result_d, sizeof(float));

    // Copy data to device
    cudaMemcpy(X_d, X_h, m * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Y_d, Y_h, n * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(result_d, &result_h, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (m + n + threadsPerBlock - 1) / threadsPerBlock;
    computeMMD<<<blocks, threadsPerBlock>>>(X_d, Y_d, m, n, dim, sigma, result_d);

    // Copy result back to host
    cudaMemcpy(&result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(result_d);

    // Print result
    std::cout << "MMD value: " << result_h << std::endl;

    return 0;
}
