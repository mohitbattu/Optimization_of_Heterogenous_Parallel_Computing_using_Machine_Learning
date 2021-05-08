#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <functional>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <chrono>
#include <list>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
using namespace std;
#define ROW_TILE_WIDTH  64
#define COL_TILE_WIDTH  64

#define EPSILON         (1e-6)

template<typename T>
__global__
void naive_matrix_multiply(T* A, T* B, T* C, int width, int cRows, int cCols)
{
    __shared__ T shATile[ROW_TILE_WIDTH][COL_TILE_WIDTH];
    __shared__ T shBTile[ROW_TILE_WIDTH][COL_TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T pValue = 0;

    // iterate for width/COL_TILE_WIDTH number of times
    // to compute the C tile
    for (int p = 0; p < width / COL_TILE_WIDTH; p++) {
        //load values to tiles from A and B
        shATile[threadIdx.y][threadIdx.x] = A[row * width + p * ROW_TILE_WIDTH + threadIdx.x];
        shBTile[threadIdx.y][threadIdx.x] = B[(p * COL_TILE_WIDTH + threadIdx.y) * cCols + col];

        // wait until all threads finish loading values
        __syncthreads();
        // update pValue for this thread
        for (int i = 0; i < COL_TILE_WIDTH; i++) pValue += shATile[threadIdx.y][i] * shBTile[i][threadIdx.x];
        // wait until all threads finish computing pValue before overwriting
        __syncthreads();

    }

    C[row * cCols + col] = pValue;

}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float()> F) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i * cols + j] = F();
        }
    }
}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float(int, int)> F) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i * cols + j] = F(i, j);
        }
    }
}

template<typename T>
void print_matrix(T* M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << M[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}



template<typename T>
void naive_matrix_multiply_cpu(T* A, T* B, T* C, int width, int C_rows, int C_cols) {

    for (int i = 0; i < C_rows; i++)
        for (int j = 0; j < C_cols; j++) {
            T value = 0.0f;
            for (int k = 0; k < width; k++) {
                value += A[i * width + k] * B[k * C_cols + j];
            }


            C[i * C_cols + j] = value;
        }
}

template<typename T>
T maxDiff(T* A1, T* A2, int rows, int cols) {
    T maxDiff = A1[0] - A2[0];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            T diff = abs(A1[i * cols + j] - A2[i * cols + j]);
            if (diff > maxDiff) {
                maxDiff = diff;
            }
        }
    }


    return maxDiff;
}


int main(void)
{
    FILE* fp;
    fp = fopen("Naive_Multiplication_CPU_Mohit.csv", "w+");
    fprintf(fp, "Algorithm_Name,Input_Dimensions,Execution_Time(ms)");
    int max_limit = 800000;
    int my_list[] = {16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008, 1040, 1072, 1104, 1136, 1168, 1200, 1232, 1264, 1296, 1328, 1360, 1392, 1424, 1456, 1488, 1520, 1552, 1584, 1616, 1648, 1680, 1712, 1744, 1776, 1808, 1840, 1872, 1904, 1936, 1968, 2000, 2032, 2064, 2096, 2128, 2160, 2192, 2224, 2256, 2288, 2320, 2352, 2384, 2416, 2448, 2480, 2512, 2544, 2576, 2608, 2640, 2672, 2704, 2736, 2768, 2800, 2832, 2864, 2896, 2928, 2960, 2992, 3024, 3056, 3088, 3120, 3152, 3184, 3216, 3248, 3280, 3312, 3344, 3376, 3408, 3440, 3472, 3504, 3536, 3568, 3600, 3632, 3664, 3696, 3728, 3760, 3792, 3824, 3856, 3888, 3920, 3952, 3984, 4016, 4080, 4144, 4208, 4272, 4336, 4400, 4464, 4528, 4592, 4656, 4720, 4784, 4848, 4912,4976, 5040, 5104, 5168, 5232, 5296, 5360, 5424, 5488, 5552, 5616, 5680, 5744, 5808, 5872, 5936, 6000, 6064, 6128, 6192, 6256, 6320, 6384, 6448, 6512, 6576, 6640, 6704, 6768, 6832, 6896, 6960, 7024, 7088, 7152, 7216, 7280, 7344, 7408, 7472, 7536, 7600, 7664, 7728, 7792, 7856, 7920, 7984, 8048, 8112, 8176, 8240, 8304, 8368, 8432, 8496, 8560, 8624, 8688, 8752, 8816, 8880, 8944, 9008, 9072, 9136, 9200, 9264, 9328, 9392, 9456, 9520, 9584, 9648, 9712, 9776, 9840, 9904, 9968, 10032, 10096, 10160, 10224, 10288, 10352, 10416, 10480, 10544, 10608, 10672, 10736, 10800, 10864, 10928, 10992};
    int length = sizeof(my_list) / sizeof(my_list[0]);
    printf("%d", length);
    char algoname[100] = "naive_matrix_cpu";
    for (int i = 0; i < length; i++){
        if(my_list[i]<=1360){
        double l[5];
        int matSize = my_list[i];
        cout<<"Matrix Size : "<<matSize<<endl;
        int A_rows = matSize;
        int A_cols = matSize;
        int B_rows = matSize;
        int B_cols = matSize;
        int C_rows = A_rows;
        int C_cols = B_cols;
        int A_size = A_rows * A_cols;
        int B_size = B_rows * B_cols;
        int C_size = C_rows * C_cols;
        float* A, * B, * C, * C_cpu;
        // // timing
        // cudaEvent_t start_gpu, stop_gpu;
        // float gpu_time_ms = 0;
        // cudaEventCreate(&start_gpu);
        // cudaEventCreate(&stop_gpu);

        // Allocate Unified Memory – accessible from CPU or GPU
        cudaMallocManaged(&A, A_size * sizeof(float));
        cudaMallocManaged(&B, B_size * sizeof(float));
        cudaMallocManaged(&C, C_size * sizeof(float));
        cudaMallocManaged(&C_cpu, C_size * sizeof(float));

        // initialize A and B matrices
        auto all_ones = []() -> float {
            return 1.0f;
        };

        srand(time(NULL));
        auto rand_numbers = []() -> float {
            return static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1000));
        };
        
        auto index_based = [](int i, int j) -> float {
            return j;
        };

        initialize_matrix<float>(A, A_rows, A_cols, rand_numbers);
        initialize_matrix<float>(B, B_rows, B_cols, rand_numbers);


        // launch kernel

        dim3 dim_grid(C_cols / COL_TILE_WIDTH, C_rows / ROW_TILE_WIDTH, 1);
        dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

        //cudaEventRecord(start_gpu);
        //naive_matrix_multiply<float> << <dim_grid, dim_block >> > (A, B, C, A_cols, C_rows, C_cols);
        //cudaEventRecord(stop_gpu);

        //// Wait for GPU to finish before accessing on host
        //cudaDeviceSynchronize();

        //cudaEventSynchronize(stop_gpu);
        //cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);

        if(my_list[i]<=768){
        
        for(int j=0;j<5;j++){
        // check results on CPU
        auto t1 = std::chrono::system_clock::now();
        naive_matrix_multiply_cpu<float>(A, B, C_cpu, A_cols, C_rows, C_cols);
        auto t2 = std::chrono::system_clock::now();

        if (fabsf(maxDiff<float>(C, C_cpu, C_rows, C_cols)) <= (float)EPSILON)
            std::cout << "PASS" << std::endl;
        else
            std::cout << "FAIL" << std::endl;

        auto cpu_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
        // std::cout << "GPU time = " << gpu_time_ms << "ms" << std::endl;
        l[j]=cpu_time_ms;    
    }
    double avg;
	avg=(l[0]+l[1]+l[2]+l[3]+l[4])/5;
		cout << "Using Milliseconds Clock: (AVG)"<< endl;
		cout << " CPU time taken to execute for strassen matrices of size - " 
		<< matSize << " : " <<avg<<" ms"<< endl;
		cout << endl;
        fprintf(fp,"\n%s,%d,%lf",algoname,matSize,avg);
    }else{
        auto t1 = std::chrono::system_clock::now();
        naive_matrix_multiply_cpu<float>(A, B, C_cpu, A_cols, C_rows, C_cols);
        auto t2 = std::chrono::system_clock::now();
        if (fabsf(maxDiff<float>(C, C_cpu, C_rows, C_cols)) <= (float)EPSILON)
            std::cout << "PASS" << std::endl;
        else
            std::cout << "FAIL" << std::endl;
        auto cpu_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f;
        cout << "Using Milliseconds Clock:(else above 1000) "<< endl;
		cout << " CPU time taken to execute for strassen matrices of size - " 
		<< matSize << " : " <<cpu_time_ms<<" ms"<< endl;
		cout << endl;
        fprintf(fp,"\n%s,%d,%lf",algoname,matSize,cpu_time_ms);
    }
       
        // std::cout << "Speedup = " << cpu_time_ms / gpu_time_ms << std::endl;

  
        // fprintf(fp, "\n%s,%d,%lf", algoname, matSize, cpu_time_ms);

        
        // Free memory
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }
    else
      {
           cout<<"MAX LIMIT"<<endl;
           fprintf(fp, "\n%s,%d,%d", algoname, my_list[i], max_limit);
     }
 }
    return 0;
}