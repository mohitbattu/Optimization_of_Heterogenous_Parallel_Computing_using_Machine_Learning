#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
using namespace std;
#include "utils.h"
#include "cudaTimer.h"
#include <iostream>

#define BLOCK_SIZE 16
#define N_TEST 50
#define DEVICE_ID 0
#define MAX_DEPTH 20
const int N = 6;

float *h_A, *h_B, *matmulRef, *cublasRef;
float *d_A[MAX_DEPTH], *d_B[MAX_DEPTH], *d_C[MAX_DEPTH];
float *d_M1[MAX_DEPTH], *d_M2[MAX_DEPTH], *d_M3[MAX_DEPTH], *d_M4[MAX_DEPTH], *d_M5[MAX_DEPTH], *d_M6[MAX_DEPTH], *d_M7[MAX_DEPTH];


template <typename ring>
void fillMatrix(ring* arr, const int N)
{
	for (int i = 0; i < N; ++i)
	{
		arr[i] = (ring) (rand() & 0xF);
	}
}
/**Matrix Multiplication as the normal algorithm*/

template<typename T>
void Matrix_Multiply(T A[][N], T B[][N], T C[][N]) {  //Calculating A*B->C
     for(int i=0; i<2; i++) {
        for(int j=0; j<2; j++) {
           C[i][j] = 0;      
           for(int t=0; t<2; t++) {
              C[i][j] = C[i][j] + A[i][t]*B[t][j];        
           }  
        }        
     }
}

/**Matrix Addition*/
template <typename T>
void Matrix_Add(int n, T X[][N], T Y[][N], T Z[][N]) {
     for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
           Z[i][j] = X[i][j] + Y[i][j];        
        }        
     }     
}

template<typename T>
void Strassen(int n, T A[][N], T B[][N], T C[][N]); 

template<typename T>
void input(int n, T p[][N]);

template<typename T>
void output(int n, T C[][N]);
/**Matrix Subtraction*/
template <typename T>
void Matrix_Sub(int n, T X[][N], T Y[][N], T Z[][N]) {
     for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
           Z[i][j] = X[i][j] - Y[i][j];        
        }        
     }     
}

template<typename T>
void output(int n, T C[][N]) {
     cout<<"The Output Matrix is :"<<endl;
     for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
           cout<<C[i][j]<<" "<<endl;        
        }        
     }     
}
template <typename T> 
void Strassen(int n, T A[][N], T B[][N], T C[][N]) {
     T A11[N][N], A12[N][N], A21[N][N], A22[N][N];
     T B11[N][N], B12[N][N], B21[N][N], B22[N][N];     
     T C11[N][N], C12[N][N], C21[N][N], C22[N][N];
     T M1[N][N], M2[N][N], M3[N][N], M4[N][N], M5[N][N], M6[N][N], M7[N][N];
     T AA[N][N], BB[N][N];
     
     if(n == 2) {  //2-order
        Matrix_Multiply(A, B, C);     
     } else {
        //将矩阵A和B分成阶数相同的四个子矩阵，即分治思想。
        for(int i=0; i<n/2; i++) {
           for(int j=0; j<n/2; j++) {
              A11[i][j] = A[i][j];
              A12[i][j] = A[i][j+n/2];
              A21[i][j] = A[i+n/2][j];
              A22[i][j] = A[i+n/2][j+n/2];
              
              B11[i][j] = B[i][j];
              B12[i][j] = B[i][j+n/2];
              B21[i][j] = B[i+n/2][j];
              B22[i][j] = B[i+n/2][j+n/2];    
           }        
        }  
        
        //Calculate M1 = (A0 + A3) × (B0 + B3)
        Matrix_Add(n/2, A11, A22, AA);
        Matrix_Add(n/2, B11, B22, BB);
        Strassen(n/2, AA, BB, M1);
        
        //Calculate M2 = (A2 + A3) × B0
        Matrix_Add(n/2, A21, A22, AA);
        Strassen(n/2, AA, B11, M2);
        
        //Calculate M3 = A0 × (B1 - B3)
        Matrix_Sub(n/2, B12, B22, BB);
        Strassen(n/2, A11, BB, M3);
        
        //Calculate M4 = A3 × (B2 - B0)
        Matrix_Sub(n/2, B21, B11, BB);
        Strassen(n/2, A22, BB, M4);
        
        //Calculate M5 = (A0 + A1) × B3
        Matrix_Add(n/2, A11, A12, AA);
        Strassen(n/2, AA, B22, M5);
        
        //Calculate M6 = (A2 - A0) × (B0 + B1)
        Matrix_Sub(n/2, A21, A11, AA);
        Matrix_Add(n/2, B11, B12, BB);
        Strassen(n/2, AA, BB, M6);
        
        //Calculate M7 = (A1 - A3) × (B2 + B3)
        Matrix_Sub(n/2, A12, A22, AA);
        Matrix_Add(n/2, B21, B22, BB);
        Strassen(n/2, AA, BB, M7);
        
        //Calculate C0 = M1 + M4 - M5 + M7
        Matrix_Add(n/2, M1, M4, AA);
        Matrix_Sub(n/2, M7, M5, BB);
        Matrix_Add(n/2, AA, BB, C11);
        
        //Calculate C1 = M3 + M5
        Matrix_Add(n/2, M3, M5, C12);
        
        //Calculate C2 = M2 + M4
        Matrix_Add(n/2, M2, M4, C21);
        
        //Calculate C3 = M1 - M2 + M3 + M6
        Matrix_Sub(n/2, M1, M2, AA);
        Matrix_Add(n/2, M3, M6, BB);
        Matrix_Add(n/2, AA, BB, C22);
        
        //Set the result to C[][N]
        for(int i=0; i<n/2; i++) {
           for(int j=0; j<n/2; j++) {
              C[i][j] = C11[i][j];
              C[i][j+n/2] = C12[i][j];
              C[i+n/2][j] = C21[i][j];
              C[i+n/2][j+n/2] = C22[i][j];        
           }        
        }
     }
}

template <typename ring>
void checkResult(ring* lhs, ring* rhs, const int dim, const char* name)
{
	double max_diff = 0;
	double avg_diff = 0;
	int max_idx = 0;

	for (int i = 0; i < dim*dim; ++i)
	{
		double curr_diff = abs(lhs[i] - rhs[i]);
		avg_diff += curr_diff;
		if (curr_diff > max_diff)
		{
			max_diff = curr_diff;
			max_idx = i;
		}
	}
	avg_diff /= (dim*dim);

	printf("[%s] Avg diff is %.8lf. Max diff is %.8lf at index %d.\n", name, avg_diff, max_diff, max_idx);
}


template <typename ring>
__global__
void classicalMatmul(ring* A, ring* B, ring* C, const int dim)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int gd = gridDim.x;

	__shared__ ring _A[BLOCK_SIZE][BLOCK_SIZE], _B[BLOCK_SIZE][BLOCK_SIZE];

	if (row < dim && col < dim)
	{
		ring sum = 0;
		for (int k = 0; k < gd; ++k)
		{
			_A[threadIdx.y][threadIdx.x] = A[row*dim + k*BLOCK_SIZE + threadIdx.x];
			_B[threadIdx.y][threadIdx.x] = B[(k*BLOCK_SIZE+threadIdx.y) * dim + col];
			__syncthreads();

			for (int l = 0; l < BLOCK_SIZE; ++l)
			{
				sum += _A[threadIdx.y][l] * _B[l][threadIdx.x];
			}
			__syncthreads();
		}

		C[row*dim + col] = sum;
	}
}


template <typename ring>
void strassenMatmul(cublasHandle_t& handle, ring* A, ring* B, ring* C, const int dim, const int d, const int threshold)
{
	const int dim_2 = dim/2;

	int lda = dim, ldb = dim, ldc = dim_2;
	int m = dim_2, n = dim_2;
	ring one = 1, zero = 0, m_one = -1;

	if (dim <= threshold)
	{
		dim3 block(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid((dim+BLOCK_SIZE-1)/BLOCK_SIZE, (dim+BLOCK_SIZE-1)/BLOCK_SIZE);
		classicalMatmul<ring><<< grid, block >>>(A, B, C, dim);
		// cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &one, B, dim, A, dim, &zero, C, dim);
		return;
	}


	/* M1 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &one, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &one, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M1[d+1], dim_2, d+1, threshold);

	/* M2 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2*dim, lda, &one, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &zero, B, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M2[d+1], dim_2, d+1, threshold);

	/* M3 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &zero, A, ldb, d_A[d+1], ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2, lda, &m_one, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M3[d+1], dim_2, d+1, threshold);

	/* M4 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2*dim+dim_2, lda, &zero, A, ldb, d_A[d+1], ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2*dim, lda, &m_one, B, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M4[d+1], dim_2, d+1, threshold);

	/* M5 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, lda, &one, A+dim_2, ldb, d_A[d+1], ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2*dim+dim_2, lda, &zero, B, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M5[d+1], dim_2, d+1, threshold);

	/* M6 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2*dim, lda, &m_one, A, ldb, d_A[d+1], ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B, lda, &one, B+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M6[d+1], dim_2, d+1, threshold);

	/* M7 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A+dim_2, lda, &m_one, A+dim_2*dim+dim_2, ldb, d_A[d+1], ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, B+dim_2*dim, lda, &one, B+dim_2*dim+dim_2, ldb, d_B[d+1], ldc);
	strassenMatmul(handle, d_A[d+1], d_B[d+1], d_M7[d+1], dim_2, d+1, threshold);


	/* C1 */
	lda = dim, ldb = dim/2, ldc = dim;  // C = C + B
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C, lda, &one, d_M1[d+1], ldb, C, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &one, d_M4[d+1], ldb, C, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &m_one, d_M5[d+1], ldb, C, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C, lda, &one, d_M7[d+1], ldb, C, ldc);

	/* C2 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C+dim_2, lda, &one, d_M3[d+1], ldb, C+dim_2, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2, lda, &one, d_M5[d+1], ldb, C+dim_2, ldc);

	/* C3 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C+dim_2*dim, lda, &one, d_M2[d+1], ldb, C+dim_2*dim, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim, lda, &one, d_M4[d+1], ldb, C+dim_2*dim, ldc);

	/* C4 */
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, C+dim_2*dim+dim_2, lda, &one, d_M1[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim+dim_2, lda, &m_one, d_M2[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim+dim_2, lda, &one, d_M3[d+1], ldb, C+dim_2*dim+dim_2, ldc);
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, C+dim_2*dim+dim_2, lda, &one, d_M6[d+1], ldb, C+dim_2*dim+dim_2, ldc);
}


void cublasMatmul(cublasHandle_t& handle, const float* A, const float* B, float* C, const int dim)
{
	int lda = dim, ldb = dim, ldc = dim;
	const int m = dim, n = dim, k = dim;
	const float a = 1;
	const float b = 0;
	const float *alpha = &a;
	const float *beta = &b;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc);
}


int main(void)
{
	


	/* Initialize */

	int nDim = 20;
	int threshold = 18;
	int check = 3;
	int A[N][N],B[N][N],C[N][N]; 
for(int i=0; i<N; i++) {
	for(int j=0; j<N; j++) {
	   A[i][j] = i * j;
	   B[i][j] = i * j;   
	}        
 }

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nDim+BLOCK_SIZE-1)/BLOCK_SIZE, (nDim+BLOCK_SIZE-1)/BLOCK_SIZE);

	assert(nDim >= threshold && threshold >= BLOCK_SIZE);

	setDeviceAndGetInfo(DEVICE_ID);

	size_t nBytes = nDim * nDim * sizeof(float);

	h_A = (float*) malloc(nBytes);
	h_B = (float*) malloc(nBytes);
	matmulRef = (float*) malloc(nBytes);
	cublasRef = (float*) malloc(nBytes);

	srand(0);
	fillMatrix<float>(h_A, nDim*nDim);
	fillMatrix<float>(h_B, nDim*nDim);

	int depth, _dim = nDim;
	for (depth = 0; depth < MAX_DEPTH && _dim > 0; ++depth)
	{
		cudaMalloc((float**) &d_A[depth], _dim*_dim*sizeof(float));
		cudaMalloc((float**) &d_B[depth], _dim*_dim*sizeof(float));

		if (depth == 0)
		{
			cudaMalloc((float**) &d_C[depth], _dim*_dim*sizeof(float));
		}
		else
		{
			cudaMalloc((float**) &d_M1[depth], _dim*_dim*sizeof(float));
			cudaMalloc((float**) &d_M2[depth], _dim*_dim*sizeof(float));
			cudaMalloc((float**) &d_M3[depth], _dim*_dim*sizeof(float));
			cudaMalloc((float**) &d_M4[depth], _dim*_dim*sizeof(float));
			cudaMalloc((float**) &d_M5[depth], _dim*_dim*sizeof(float));
			cudaMalloc((float**) &d_M6[depth], _dim*_dim*sizeof(float));
			cudaMalloc((float**) &d_M7[depth], _dim*_dim*sizeof(float));
		}
		_dim /= 2;
	}

	cudaMemcpy(d_A[0], h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B[0], h_B, nBytes, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);

	CudaTimer ct;


	/* Prepare result */

	if (check)
	{
		cublasMatmul(handle, d_A[0], d_B[0], d_C[0], nDim);
		cudaMemcpy(cublasRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
	}


	/* Run classicalMatmul */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		classicalMatmul<float><<< grid, block >>>(d_A[0], d_B[0], d_C[0], nDim);
		cudaDeviceSynchronize();
	}
	ct.stop();
	printf("[classicalMatmul] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(matmulRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
		checkResult<float>(cublasRef, matmulRef, nDim, "classicalMatmul");
	}

	/* Run strassenMatmul */

	ct.start();
	for (int i = 0; i < N_TEST; ++i)
	{
		strassenMatmul<float>(handle, d_A[0], d_B[0], d_C[0], nDim, 0, threshold);
	}
	ct.stop();

	cudaMemcpy(matmulRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
	printf("[strassenMatmul] %.5fms\n", ct.value()/N_TEST);

	if (check)
	{
		cudaMemcpy(matmulRef, d_C[0], nBytes, cudaMemcpyDeviceToHost);
		checkResult<float>(cublasRef, matmulRef, nDim, "strassenMatmul");
	}


	//RUN CPU
	if (check){
	clock_t begin = clock();
	Strassen(N,A,B,C);
	output(N,C);
	clock_t end = clock();
	double time_spent = (double)1000*(end - begin) / CLOCKS_PER_SEC;
	printf("CPU time= %lf ms\n", time_spent);
	}
	/* Free memory */

	cublasDestroy(handle);

	for (int i = 0; i < depth; ++i)
	{
		cudaFree(d_A[i]);
		cudaFree(d_B[i]);

		if (i == 0)
		{
			cudaFree(d_C[i]);
		}
		else
		{
			cudaFree(d_M1[i]);
			cudaFree(d_M2[i]);
			cudaFree(d_M3[i]);
			cudaFree(d_M4[i]);
			cudaFree(d_M5[i]);
			cudaFree(d_M6[i]);
			cudaFree(d_M7[i]);
		}
	}

	cudaDeviceReset();

	free(h_A);
	free(h_B);
	free(matmulRef);
	free(cublasRef);

	printf("Done.\n");

	return 0;
}
