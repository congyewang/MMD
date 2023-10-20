#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel to compute the RBF kernel between two sets of points
__global__ void rbf_kernel(float *X, float *Y, int nx, int ny, float sigma, float *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        float sum = 0.0;
        for (int j = 0; j < ny; j++) {
            float d = 0.0;
            for (int k = 0; k < 3; k++) {
                float diff = X[i * 3 + k] - Y[j * 3 + k];
                d += diff * diff;
            }
            sum += expf(-d / (2 * sigma * sigma));
        }
        result[i] = sum / ny;
    }
}

// Function to compute the Maximum Mean Discrepancy (MMD) between two sets of points
float compute_mmd(float *X, float *Y, int nx, int ny, float sigma) {
    float *d_X, *d_Y, *d_result;
    cudaMalloc(&d_X, nx * 3 * sizeof(float));
    cudaMalloc(&d_Y, ny * 3 * sizeof(float));
    cudaMalloc(&d_result, nx * sizeof(float));

    cudaMemcpy(d_X, X, nx * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, ny * 3 * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (nx + block_size - 1) / block_size;
    rbf_kernel<<<grid_size, block_size>>>(d_X, d_Y, nx, ny, sigma, d_result);

    float *result = (float *)malloc(nx * sizeof(float));
    cudaMemcpy(result, d_result, nx * sizeof(float), cudaMemcpyDeviceToHost);

    float mmd = 0.0;
    for (int i = 0; i < nx; i++) {
        mmd += result[i];
    }
    mmd /= nx;

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_result);
    free(result);

    return mmd;
}

int main() {
    // Test data: two sets of 3D points
    float X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    float Y[] = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0};
    int nx = 3;
    int ny = 3;
    float sigma = 1.0;

    float mmd = compute_mmd(X, Y, nx, ny, sigma);
    printf("Maximum Mean Discrepancy: %f\n", mmd);

    return 0;
}
