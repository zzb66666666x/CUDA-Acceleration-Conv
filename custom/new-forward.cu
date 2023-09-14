#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// test: rai -p . --queue rai_amd64_exclusive
// debug: rai -p . --queue rai_amd64_ece408

#define TILE_WIDTH 16

#define STREAM_WIDTH 20
#define NUM_STREAM 5

#define MAX_KERNEL_ELEMENTS 3136
__constant__ float kc[MAX_KERNEL_ELEMENTS];

// some always on optimization
#define USE_CONST_MEM 1
// choose kernel algorithm (one-hot)
#define USE_CONV 1
#define USE_MATMUL_CONV 0
// pipeline method
#define USE_STREAM 0
// optimize within classic conv kernel
#define USE_SHARE_MEM_CONV 0
#define USE_MATMUL_CONV_FUSION 1

float * cur_host_x;
float * cur_host_y;

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k4d_c(i3, i2, i1, i0) kc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define x4d_idx(i3, i2, i1, i0) ( (i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0 )
#define y4d_idx(i3, i2, i1, i0) ( (i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0 )

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int W_grid = ceil((W_out)/(float)TILE_WIDTH);
    int b = blockIdx.x;
    int m = blockIdx.y;
    int row = blockIdx.z/W_grid * TILE_WIDTH + ty;
    int col = blockIdx.z%W_grid * TILE_WIDTH + tx;

    if(row < H_out && col < W_out){
        // iterate over batch
        float res = 0;
        int idx = y4d_idx(b, m, row, col); 
        // iterate over input tensor and kernel
        for (int c=0; c<C; c++){
            for (int krow = 0; krow<K; krow++){
                for (int kcol = 0; kcol<K; kcol++){
#if(USE_CONST_MEM == 1)
                    res += (x4d(b, c, row+krow, col+kcol)*k4d_c(m, c, krow, kcol));
#else
                    res += (x4d(b, c, row+krow, col+kcol)*k4d(m, c, krow, kcol));
#endif
                }
            }
        }
        y[idx] = res;
    }

#undef y4d
#undef x4d
#undef k4d
#undef k4d_c
#undef x4d_idx
#undef y4d_idx
}

__global__ void conv_forward_kernel_share(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define k4d_c(i3, i2, i1, i0) kc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
// #define x4d_idx(i3, i2, i1, i0) ( (i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0 )
// #define y4d_idx(i3, i2, i1, i0) ( (i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0 )

    // Insert your GPU convolution kernel code here
    extern __shared__ float x_tile[]; // (TILE+K-1)*(TILE+K-1)*C
    const int x_tile_width = blockDim.x;
#define xtile(i2, i1, i0) x_tile[(i2) * (x_tile_width * x_tile_width) + (i1) * (x_tile_width) + i0]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int W_grid = ceil((W_out)/(float)TILE_WIDTH);
    int blockIdx_y = blockIdx.z/W_grid;
    int blockIdx_x = blockIdx.z%W_grid;
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h = blockIdx_y * TILE_WIDTH + ty;
    int w = blockIdx_x * TILE_WIDTH + tx;
    if (h < H && w < W){
        for (int c=0; c<C; c++){
            xtile(c, ty, tx) = x4d(b, c, h, w);
        } 
    }else{
        for (int c=0; c<C; c++){
            xtile(c, ty, tx) = 0.0f;
        } 
    }
    __syncthreads();
    if (ty < TILE_WIDTH && tx < TILE_WIDTH){
        float res = 0.0f;
        for (int c=0; c<C; c++){
            for (int krow = 0; krow<K; krow++){
                for (int kcol = 0; kcol<K; kcol++){
#if(USE_CONST_MEM == 1)
                    res += (k4d_c(m, c, krow, kcol) * xtile(c, krow+ty, kcol+tx));
#else
                    res += (k4d(m, c, krow, kcol) * xtile(c, krow+ty, kcol+tx));
#endif
                }
            }
        }
        if (h < H_out && w < W_out){
            y4d(b, m, h, w) = res;
        }
    }
#undef y4d
#undef x4d
#undef k4d
#undef k4d_c
// #undef x4d_idx
// #undef y4d_idx
#undef xtile
}

__global__ void conv_forward_unroll(const float* x, float* x_unroll, const int B, const int C, const int H, const int W, const int K){
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int x_num_rows = K*K*C;
    int x_num_cols = W_out * H_out;
    int x_size = x_num_cols * x_num_rows;
    int b = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int w = bx * TILE_WIDTH + tx;
    int h = by * TILE_WIDTH + ty;
    if (w < W_out && h < H_out && b < B){
        for (int c=0; c < C; c++){
            int w_base = c * K * K;
            for (int p=0; p<K; p++){
                for (int q=0; q<K; q++){
                    int h_unroll = w_base + p*K + q;
                    int w_unroll = h * W_out + w;
                    x_unroll[b * x_size + h_unroll * x_num_cols + w_unroll] = x4d(b, c, h+p, w+q);
                }
            }
        }
    }

// single thread
    // int H_out = H-K+1;
    // int W_out = W-K+1;
    // int x_size = K*K*C*W_out*H_out;
    // for (int b=0; b<B; b++){
    //     for (int c=0; c<C; c++){
    //         int w_base = c*K*K;
    //         for (int p=0; p<K; p++){
    //             for (int q=0; q<K; q++){
    //                 for (int h=0; h<H_out; h++){
    //                     for (int w=0; w<W_out; w++){
    //                         int h_unroll = w_base + p*K + q;
    //                         int w_unroll = h*W_out + w;
    //                         x_unroll[b*x_size + h_unroll*W_out*H_out + w_unroll] = x4d(b,c,h+p,w+q);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
#undef x4d
}

__global__ void conv_forward_kernel_matmul(float *y, const float *x_unroll, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
#define x3d(i2, i1, i0) x_unroll[(i2) * (C*K*K*W_out*H_out) + (i1) * (W_out*H_out) + (i0)]
#define y3d(i2, i1, i0) y[(i2) * (M*W_out*H_out) + (i1) * (W_out*H_out) + (i0)]
#define k2d_c(i1, i0) kc[(i1) * (C*K*K) + (i0)]
#define k2d(i1, i0) k[(i1) * (C*K*K) + (i0)]
    int x_num_rows = K*K*C;
    int x_num_cols = W_out * H_out;
    // int x_size = x_num_cols * x_num_rows;
    // int y_size = M * x_num_cols;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batch = blockIdx.z;
    __shared__ float sub_x_unroll[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_k[TILE_WIDTH][TILE_WIDTH];
    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;
    float ans = 0.0f;
    for (int tile_idx=0; tile_idx < ceil((float)(x_num_rows)/TILE_WIDTH); tile_idx ++){
        // load kernel
        if ((Row < M) && ((tile_idx * TILE_WIDTH + tx) < x_num_rows)){
#if (USE_CONST_MEM == 1)
            sub_k[ty][tx] = k2d_c(Row, tile_idx * TILE_WIDTH + tx);
#else
            sub_k[ty][tx] = k2d(Row, tile_idx * TILE_WIDTH + tx);
#endif
        }else{
            sub_k[ty][tx] = 0.0f;
        }
        // load input feature map
        if (((tile_idx * TILE_WIDTH + ty) < x_num_rows) && (Col < x_num_cols)){
            sub_x_unroll[ty][tx] = x3d(batch, tile_idx * TILE_WIDTH + ty, Col);
        }else{
            sub_x_unroll[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int tmp=0; tmp < TILE_WIDTH; tmp++){
            ans += (sub_k[ty][tmp] * sub_x_unroll[tmp][tx]);
        }
        __syncthreads();
    }
    if (Row < M && Col < x_num_cols){
        y3d(batch, Row, Col) = ans;
    }
#undef k2d
#undef x3d
#undef y3d
}


__global__ void conv_forward_kernel_matmul_fusion(float *y, const float* x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define k2d_c(i1, i0) kc[(i1) * (C*K*K) + (i0)]
#define k2d(i1, i0) k[(i1) * (C*K*K) + (i0)]

    __shared__ float sub_x_unroll[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sub_k[TILE_WIDTH][TILE_WIDTH];

    int x_num_rows = K*K*C;
    int x_num_cols = W_out * H_out;
    int x_size = x_num_cols * x_num_rows;
    int b = blockIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;
    float ans = 0.0f;
    for (int tile_idx=0; tile_idx < ceil((float)(x_num_rows)/TILE_WIDTH); tile_idx ++){
        // load kernel
        int tile_x = tile_idx * TILE_WIDTH + tx;
        int tile_y = tile_idx * TILE_WIDTH + ty;

        if ((Row < M) && (tile_x < x_num_rows)){
#if (USE_CONST_MEM == 1)
            sub_k[ty][tx] = k2d_c(Row, tile_idx * TILE_WIDTH + tx);
#else
            sub_k[ty][tx] = k2d(Row, tile_idx * TILE_WIDTH + tx);
#endif
        }else{
            sub_k[ty][tx] = 0.0f;
        }
        // load input feature map
        int X_c = tile_y / (K*K);
        int X_p = (tile_y % (K*K))/K;
        int X_q = (tile_y % (K*K))%K;
        int X_h = Col / W_out;
        int X_w = Col % W_out;

        if ((tile_y < x_num_rows) && (Col < x_num_cols)){
            sub_x_unroll[ty][tx] = x4d(b, X_c, X_h + X_p, X_w + X_q);
        }else{
            sub_x_unroll[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int tmp=0; tmp < TILE_WIDTH; tmp++){
            ans += (sub_k[ty][tmp] * sub_x_unroll[tmp][tx]);
        }
        __syncthreads();
    }

    int Y_m = Row;
    int Y_h = Col / W_out;
    int Y_w = Col % W_out;

    if (Row < M && Col < x_num_cols){
        y4d(b, Y_m, Y_h, Y_w) = ans;
    }
#undef k2d
#undef k2d_c
#undef x4d
#undef y4d
}

__host__ void get_stream_range(int batch_num, int stream_id, int B, int* begin_ptr, int * size_ptr){
    if (batch_num + stream_id*STREAM_WIDTH > B){
        *size_ptr = 0;
    }else{
        *begin_ptr = batch_num + stream_id * STREAM_WIDTH;
        int left_num = B-(batch_num + stream_id*STREAM_WIDTH);
        *size_ptr = (left_num < STREAM_WIDTH)? left_num: STREAM_WIDTH;
    }
}

__host__ void copy_input_async(float *host_array, float* device_array, int offset, int size, cudaStream_t* stream_ptr){
    if (size == 0)
        return;
    cudaMemcpyAsync(device_array + offset, host_array + offset, size * sizeof(float), cudaMemcpyHostToDevice, *stream_ptr);
}

__host__ void conv_kernel_async(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid, int H_grid, cudaStream_t* stream_ptr){
    if (B == 0){
        // printf("pass\n");
        return;
    }
    // printf("batch %d\n", B);
    dim3 dimGrid(B, M, W_grid * H_grid);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock, 0, *stream_ptr>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}

__host__ void copy_output_async(float *host_array, float* device_array, int offset, int size, cudaStream_t* stream_ptr){
    if (size == 0)
        return;
    cudaMemcpyAsync(host_array + offset, device_array + offset, size*sizeof(float), cudaMemcpyDeviceToHost, *stream_ptr);
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    // std::cout<<"using CUDA in forward pass"<<std::endl;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int size_x = sizeof(float)*B*C*W*H;
    int size_y = sizeof(float)*B*M*W_out*H_out;
    int size_k = sizeof(float)*K*K*C*M;
    // printf("kernel size: %d\n", K*K*C*M);
    cudaMalloc((void**)device_x_ptr, size_x);
    cudaMalloc((void**)device_y_ptr, size_y);
#if (USE_CONST_MEM != 1)
    cudaMalloc((void**)device_k_ptr, size_k);
#endif

#if(USE_STREAM == 1)
    cur_host_x = (float*)host_x;
    cur_host_y = (float*)host_y;
#else
    cudaMemcpy(*device_x_ptr, host_x, size_x, cudaMemcpyHostToDevice);
#endif
   
#if (USE_CONST_MEM == 1)
    cudaMemcpyToSymbol(kc, host_k, size_k);
#else
    cudaMemcpy(*device_k_ptr, host_k, size_k, cudaMemcpyHostToDevice);
#endif
}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((W_out)/(float)TILE_WIDTH);
    int H_grid = ceil((H_out)/(float)TILE_WIDTH);
#if (USE_MATMUL_CONV == 1)
// for matmul, we can use stream
#if (USE_STREAM == 1)
#error TODO: stream with matmul version of conv 
#else
#if (USE_MATMUL_CONV_FUSION == 1)
    dim3 dimGrid_fusion(ceil(((float)(W_out * H_out))/TILE_WIDTH),ceil((float)M/TILE_WIDTH), B);
    dim3 dimBlock_fusion(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel_matmul_fusion<<<dimGrid_fusion, dimBlock_fusion>>>(device_y, device_x, device_k, B, M, C, H, W, K);
#else
    float * device_x_unroll;
    cudaMalloc((void**)&device_x_unroll, B*H_out*W_out*K*K*C*sizeof(float));
    dim3 dimGrid_unroll(ceil(W_out/(float)TILE_WIDTH), ceil(H_out/(float)TILE_WIDTH), B);
    dim3 dimBlock_unroll(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 dimGrid_unroll(1, 1, 1);
    // dim3 dimBlock_unroll(1, 1, 1);
    conv_forward_unroll<<<dimGrid_unroll, dimBlock_unroll>>>(device_x, device_x_unroll, B, C, H, W, K);
    dim3 dimGrid(ceil(((float)(W_out * H_out))/TILE_WIDTH),ceil((float)M/TILE_WIDTH), B);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel_matmul<<<dimGrid, dimBlock>>>(device_y, device_x_unroll, device_k, B, M, C, H, W, K);
    cudaFree(device_x_unroll);
#endif
#endif
#endif

#if (USE_CONV == 1)
// it's also possible for classic conv kernel to benefit from stream
#if(USE_STREAM == 1)
#if (USE_SHARE_MEM_CONV == 1)
#error TODO: stream with classic method of conv using shared memory
#else
    cudaStream_t stream0, stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    int x_size_each = C*W*H;
    int y_size_each = M*W_out*H_out;
    for (int batch_num = 0; batch_num < B; batch_num += (STREAM_WIDTH * NUM_STREAM)){
        int size0, size1, size2, size3, size4;
        int begin0, begin1, begin2, begin3, begin4;
        get_stream_range(batch_num, 0, B, &begin0, &size0);
        get_stream_range(batch_num, 1, B, &begin1, &size1);
        get_stream_range(batch_num, 2, B, &begin2, &size2);
        get_stream_range(batch_num, 3, B, &begin3, &size3);
        get_stream_range(batch_num, 4, B, &begin4, &size4);
        // printf("%d, %d, %d, %d\n", size0, size1, size2, size3);
        // copy inside: float *host_array, float* device_array, int offset_bytes, int size_bytes, cudaStream_t* stream_ptr
        copy_input_async(cur_host_x, (float*)device_x, begin0 * x_size_each, size0 * x_size_each, &stream0);
        copy_input_async(cur_host_x, (float*)device_x, begin1 * x_size_each, size1 * x_size_each, &stream1);
        copy_input_async(cur_host_x, (float*)device_x, begin2 * x_size_each, size2 * x_size_each, &stream2);
        copy_input_async(cur_host_x, (float*)device_x, begin3 * x_size_each, size3 * x_size_each, &stream3);
        copy_input_async(cur_host_x, (float*)device_x, begin4 * x_size_each, size4 * x_size_each, &stream4);
        // kernel: float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K, dim3 dimGrid, dim3 dimBlock
        conv_kernel_async(device_y + begin0 * y_size_each, device_x + begin0 * x_size_each, device_k, size0, M, C, H, W, K, W_grid, H_grid, &stream0);
        conv_kernel_async(device_y + begin1 * y_size_each, device_x + begin1 * x_size_each, device_k, size1, M, C, H, W, K, W_grid, H_grid, &stream1);
        conv_kernel_async(device_y + begin2 * y_size_each, device_x + begin2 * x_size_each, device_k, size2, M, C, H, W, K, W_grid, H_grid, &stream2);
        conv_kernel_async(device_y + begin3 * y_size_each, device_x + begin3 * x_size_each, device_k, size3, M, C, H, W, K, W_grid, H_grid, &stream3);
        conv_kernel_async(device_y + begin4 * y_size_each, device_x + begin4 * x_size_each, device_k, size4, M, C, H, W, K, W_grid, H_grid, &stream4);
        // copy out: float *host_array, float* device_array, int offset, int size, cudaStream_t* stream_ptr
        copy_output_async(cur_host_y, (float*)device_y, begin0 * y_size_each, size0 * y_size_each, &stream0);
        copy_output_async(cur_host_y, (float*)device_y, begin1 * y_size_each, size1 * y_size_each, &stream1);
        copy_output_async(cur_host_y, (float*)device_y, begin2 * y_size_each, size2 * y_size_each, &stream2);
        copy_output_async(cur_host_y, (float*)device_y, begin3 * y_size_each, size3 * y_size_each, &stream3);
        copy_output_async(cur_host_y, (float*)device_y, begin4 * y_size_each, size4 * y_size_each, &stream4);
    }

#endif

#else
#if (USE_SHARE_MEM_CONV == 1)
    dim3 dimGrid(B, M, W_grid * H_grid);
    dim3 dimBlock(TILE_WIDTH + K - 1, TILE_WIDTH + K - 1, 1);
    conv_forward_kernel_share<<<dimGrid, dimBlock, sizeof(float)*(TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)*C>>>(device_y, device_x, device_k, B, M, C, H, W, K);
#else
    dim3 dimGrid(B, M, W_grid * H_grid);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
#endif
#endif
#endif
    cudaDeviceSynchronize();

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // int size_x = sizeof(float)*B*C*W*H;
    int size_y = sizeof(float)*B*M*W_out*H_out;
    // int size_k = sizeof(float)*K*K*C*M;
#if (USE_STREAM != 1)
    cudaMemcpy(host_y, device_y, size_y, cudaMemcpyDeviceToHost);
#endif

    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
#if (USE_CONST_MEM != 1)
    cudaFree(device_k);
#endif
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
