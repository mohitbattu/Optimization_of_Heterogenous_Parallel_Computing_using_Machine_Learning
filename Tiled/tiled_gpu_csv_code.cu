#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>   
#include <list>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
using namespace std;
#define BLOCK_SIZE 64

/*
*********************************************************************
function name: gpu_matrix_mult
description: dot product of two matrix (not only square)
parameters:
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C)
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

/*
*********************************************************************
function name: gpu_square_matrix_mult
description: dot product of two matrix (not only square) in GPU
parameters:
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C)
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(int* d_a, int* d_b, int* d_result, int n)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if (idx >= n * n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if (idx >= n * n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

/*
*********************************************************************
function name: gpu_matrix_transpose
description: matrix transpose
parameters:
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: cpu_matrix_mult
description: dot product of two matrix (not only square) in CPU,
             for validating GPU results
parameters:
            &a CPU host pointer to a m X n matrix (A)
            &b CPU host pointer to a n X k matrix (B)
            &c CPU host output purpose pointer to a m X k matrix (C)
            to store the result
return: none
*********************************************************************
*/
void cpu_matrix_mult(int* h_a, int* h_b, int* h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

/*
*********************************************************************
function name: main
description: test and compare
parameters:
            none
return: none
*********************************************************************
*/
int main(int argc, char const* argv[])
{
    FILE* fp;
    fp = fopen("Tiled_Multiplication.csv", "w+");
    fprintf(fp, "Algorithm_Name,Input_Dimensions,Execution_Time(ms)");
    int m, n, k;
    /* Fixed seed for illustration */
    srand(3333);/*
    printf("please type in m n and k\n");
    scanf("%d %d %d", &m, &n, &k);*/
    // allocate memory in host RAM, h_cc is used to store CPU result
    char algoname[100] = "Tiled_Matrix_Multiplication_GPU";
    //char usage[20] = "GPU";
    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    int max_limit = 40;
    int my_list[] = {16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1040, 1072, 1104, 1136, 1168, 1200, 1232, 1264, 1296, 1328, 1360, 1392, 1424, 1456, 1488, 1520, 1552, 1584, 1616, 1648, 1680, 1712, 1744, 1776, 1808, 1840, 1872, 1904, 1936, 1968, 2000, 2032, 2064, 2096, 2128, 2160, 2192, 2224, 2256, 2288, 2320, 2352, 2384, 2416, 2448, 2480, 2512, 2544, 2576, 2608, 2640, 2672, 2704, 2736, 2768, 2800, 2832, 2864, 2896, 2928, 2960, 2992, 3024, 3056, 3088, 3120, 3152, 3184, 3216, 3248, 3280, 3312, 3344, 3376, 3408, 3440, 3472, 3504, 3536, 3568, 3600, 3632, 3664, 3696, 3728, 3760, 3792, 3824, 3856, 3888, 3920, 3952, 3984, 4016, 4080, 4144, 4208, 4272, 4336, 4400, 4464, 4528, 4592, 4656, 4720, 4784, 4848, 4912,4976, 5040, 5104, 5168, 5232, 5296, 5360, 5424, 5488, 5552, 5616, 5680, 5744, 5808, 5872, 5936, 6000, 6064, 6128, 6192, 6256, 6320, 6384, 6448, 6512, 6576, 6640, 6704, 6768, 6832, 6896, 6960, 7024, 7088, 7152, 7216, 7280, 7344, 7408, 7472, 7536, 7600, 7664, 7728, 7792, 7856, 7920, 7984, 8048, 8112, 8176, 8240, 8304, 8368, 8432, 8496, 8560, 8624, 8688, 8752, 8816, 8880, 8944, 9008, 9072, 9136, 9200, 9264, 9328, 9392, 9456, 9520, 9584, 9648, 9712, 9776, 9840, 9904, 9968, 10032, 10096, 10160, 10224, 10288, 10352, 10416, 10480, 10544, 10608, 10672, 10736, 10800, 10864, 10928, 10992};
    int length = sizeof(my_list) / sizeof(my_list[0]);
    printf("%d", length);
    double arr[7];
    for (int i = 0; i < length; i++) {
        m = n = k = my_list[i];

        int* h_a, * h_b, * h_c, * h_cc;
        cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
        cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
        cudaMallocHost((void**)&h_c, sizeof(int) * m * k);
        cudaMallocHost((void**)&h_cc, sizeof(int) * m * k);

        // random initialize matrix A
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                h_a[i * n + j] = rand() % 1024;
            }
        }

        // random initialize matrix B
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                h_b[i * k + j] = rand() % 1024;
            }
        }
        int matSize = my_list[i];        
/***********************************************************************/
for(int j=0;j<7;j++){
        // some events to count the execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        double startTime;
        double elapsedTime;
        double standardMean;
        double strassenMean;
        
        vector< vector<double> >
            a(matSize, vector<double>(matSize)),
            b(matSize, vector<double>(matSize)),
            c(matSize, vector<double>(matSize));
        //cout << "Here We are generating a series of inputs" << endl;        
        //initMat(a, b, matSize);
        // start to count execution time of GPU version
        cudaEventRecord(start, 0);
        int* d_a, * d_b, * d_c;
        cudaMalloc((void**)&d_a, sizeof(int) * m * n);
        cudaMalloc((void**)&d_b, sizeof(int) * n * k);
        cudaMalloc((void**)&d_c, sizeof(int) * m * k);
        // copy matrix A and B from host to device memory
        cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);

        unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
        if (m == n && n == k)
        {
            gpu_square_matrix_mult << <dimGrid, dimBlock >> > (d_a, d_b, d_c, n);

        }
        /*else
        {
            
        }*/
        // Transefr results from device to host 
        cudaMemcpy(h_c, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        // time counting terminate
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        arr[j] = gpu_elapsed_time_ms;
   
/***********************************************************************/
        // compute time elapse on GPU computing
        
        // printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);
        
        // validate results computed by GPU
        int all_ok = 1;
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < k; ++j)
            {
                //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
                if (h_cc[i * k + j] != h_c[i * k + j])
                {
                    all_ok = 0;
                }
            }
            //printf("\n");
        }
        // free memory
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
        
        cudaFreeHost(h_a);
        cudaFreeHost(h_b);
        cudaFreeHost(h_c);
        cudaFreeHost(h_cc);
        double mean = (arr[2] + arr[3] + arr[4] + arr[5] + arr[6]) / 5;
        fprintf(fp, "\n%s,%d,%lf", algoname,matSize, mean);
        printf("Time elapsed on tiled matrix multiplication of on GPU: %f ms.\n\n", mean);
    }
    return 0;
}