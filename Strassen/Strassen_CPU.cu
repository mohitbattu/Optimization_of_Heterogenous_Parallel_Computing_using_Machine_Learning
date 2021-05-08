#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <list>
using namespace std;

//size at which the sequential multiplication is used instead of recursive Strassen
int thresholdSize = 128;  

void initMat(vector< vector<double> > &a, vector< vector<double> > &b, int n)
{
		// initialize matrices and fill them with random values
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				a[i][j] = (double)rand()/RAND_MAX*10;
				b[i][j] = (double)rand()/RAND_MAX*10;
			}
		}
}
	
void multiplyMatStandard(vector< vector<double> > &a, 
		vector< vector<double> > &b, vector< vector<double> > &c, int n)
{
		// standard matrix multipmlication: C <- C + A x B
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				double temp  = 0;
				for (int k = 0; k < n; ++k) {
					temp += a[i][k] * b[k][j];
				}
				c[i][j]=temp;
			}
		}
}

int getNextPowerOfTwo(int n)
{
	return pow(2, int(ceil(log2(n))));
}

void fillZeros(vector< vector<double> > &newA, vector< vector<double> > &newB,
	       vector< vector<double> > &a, vector< vector<double> > &b, int n)
{
  //pad matrix with zeros
	for (int i=0; i<n; i++){
		for (int j=0; j<n; j++){
			newA[i][j] = a[i][j];
			newB[i][j] = b[i][j];
		}
	}
}

void add(vector< vector<double> > &a, vector< vector<double> > &b,
	 vector< vector<double> > &resultMatrix, int n)
{
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] + b[i][j];
		}
	}
}

void subtract(vector< vector<double> > &a, vector< vector<double> > &b,
	      vector< vector<double> > &resultMatrix, int n)
{
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			resultMatrix[i][j] = a[i][j] - b[i][j];
		}
	}
}
	
void multiplyStrassen(vector< vector<double> > &a,
	vector< vector<double> > &b, vector< vector<double> > &c, int n)
{	
	if(n<=thresholdSize){
		multiplyMatStandard(a, b, c, n);
	}
	else{
		//expand and fill with zeros if matrix size is not a power of two
		int newSize = getNextPowerOfTwo(n);
		vector< vector<double> > 
			newA(newSize, vector<double>(newSize)), 
			newB(newSize, vector<double>(newSize)), 
			newC(newSize, vector<double>(newSize));
		if(n==newSize){   //matrix size is already a power of two
			newA = a;
			newB = b;
		}
		else{
			fillZeros(newA, newB, a, b, n);
		}
		
		//initialize submatrices
		int blockSize = newSize/2;  //size for a partition matrix
		vector<double> block (blockSize);
		vector< vector<double> > 
			/*partitions of newA*/
			a11(blockSize, block), a12(blockSize, block), 
			a21(blockSize, block), a22(blockSize, block), 
			/*partitions of newB*/
			b11(blockSize, block), b12(blockSize, block), 
			b21(blockSize, block), b22(blockSize, block), 
			/*partitions of newC*/
			c11(blockSize, block), c12(blockSize, block), 
			c21(blockSize, block), c22(blockSize, block), 
			/*matrices storing intermediate results*/
			aBlock(blockSize, block), bBlock(blockSize, block),
			/*set of submatrices derived from partitions*/
			m1(blockSize, block), m2(blockSize, block), 
			m3(blockSize, block), m4(blockSize, block),  
			m5(blockSize, block), m6(blockSize, block), 
			m7(blockSize, block);  
		
		//partition matrices
		for (int i=0; i<blockSize; i++){
			for (int j=0; j<blockSize; j++){
				a11[i][j] = newA[i][j];
				a12[i][j] = newA[i][j+blockSize];
				a21[i][j] = newA[i+blockSize][j];
				a22[i][j] = newA[i+blockSize][j+blockSize];
				b11[i][j] = newB[i][j];
				b12[i][j] = newB[i][j+blockSize];
				b21[i][j] = newB[i+blockSize][j];
				b22[i][j] = newB[i+blockSize][j+blockSize];
			}
		}
		
		//compute submatrices
		//m1 = (a11+a22)(b11+b22)
		add(a11, a22, aBlock, blockSize);
		add(b11, b22, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m1, blockSize);
		
		//m2 = (a21+a22)b11
		add(a21, a22, aBlock, blockSize);
		multiplyStrassen(aBlock, b11, m2, blockSize);
		
		//m3 = a11(b12-b22)
		subtract(b12, b22, bBlock, blockSize);
		multiplyStrassen(a11, bBlock, m3, blockSize);
		
		//m4 = a22(b21-b11)
		subtract(b21, b11, bBlock, blockSize);
		multiplyStrassen(a22, bBlock, m4, blockSize);
		
		//m5 = (a11+a12)b22
		add(a11, a12, aBlock, blockSize);
		multiplyStrassen(aBlock, b22, m5, blockSize);
		
		//m6 = (a21-a11)(b11+b12)
		subtract(a21, a11, aBlock, blockSize);
		add(b11, b12, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m6, blockSize);
		
		//m7 = (a12-a22)(b12+b22)
		subtract(a12, a22, aBlock, blockSize);
		add(b12, b22, bBlock, blockSize);
		multiplyStrassen(aBlock, bBlock, m7, blockSize);
		
		//calculate result submatrices
		//c11 = m1+m4-m5+m7
		add(m1, m4, aBlock, blockSize);
		subtract(aBlock, m5, bBlock, blockSize);
		add(bBlock, m7, c11, blockSize);
		
		//c12 = m3+m5
		add(m3, m5, c12, blockSize);
		
		//c21 = m2+m4
		add(m2, m4, c12, blockSize);
		
		//c22 = m1-m2+m3+m6
		subtract(m1, m2, aBlock, blockSize);
		add(aBlock, m3, bBlock, blockSize);
		add(bBlock, m6, c22, blockSize);
		
		//calculate final result matrix
		for(int i=0; i<blockSize; i++){
			for(int j=0; j<blockSize; j++){
				newC[i][j] = c11[i][j];
				newC[i][blockSize+j] = c12[i][j];
				newC[blockSize+i][j] = c21[i][j];
				newC[blockSize+i][blockSize+j] = c22[i][j];
			}
		}
		
		//remove additional values from expanded matrix
		for(int i=0; i<n; i++){
			for(int j=0; j<n; j++){
				c[i][j] = newC[i][j];
			}
		}
	}
}

double calculateMean(vector<double> data, int size) 
{
    double sum = 0.0, mean = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += data[i];
    }

    mean = sum / size;
    return mean;
}
	
int main()
{
	//srand(time(0));   //seed for random number generation
	// number of sample size considered to evaluate average execution time      
	FILE * fp;
    fp=fopen("Strassen_Multiplication_UPDATED_CPU.csv","w+");
    fprintf(fp, "Algorithm_Name,Input_Dimensions,Execution_Time(ms)");
	double startTime;
	double elapsedTime;
	double standardMean;
	double strassenMean;
    float cpu_elapsed_time_ms;
    char algoname[100]="Strassen_Matrix_Multiplication_CPU";
    int my_list[]={1008, 1040, 1072, 1104, 1136, 1168, 1200, 1232, 1264, 1296, 1328, 1360, 1392, 1424, 1456, 1488, 1520, 1552, 1584, 1616, 1648, 1680, 1712, 1744, 1776, 1808, 1840, 1872, 1904, 1936, 1968, 2000, 2032, 2064, 2096, 2128, 2160, 2192, 2224, 2256, 2288, 2320, 2352, 2384, 2416, 2448, 2480, 2512, 2544, 2576, 2608, 2640, 2672, 2704, 2736, 2768, 2800, 2832, 2864, 2896, 2928, 2960, 2992, 3024, 3056, 3088, 3120, 3152, 3184, 3216, 3248, 3280, 3312, 3344, 3376, 3408, 3440, 3472, 3504, 3536, 3568, 3600, 3632, 3664, 3696, 3728, 3760, 3792, 3824, 3856, 3888, 3920, 3952, 3984, 4016, 4080, 4144, 4208, 4272, 4336, 4400, 4464, 4528, 4592, 4656, 4720, 4784, 4848, 4912, 4976, 5040, 5104, 5168, 5232, 5296, 5360, 5424, 5488, 5552, 5616, 5680, 5744, 5808, 5872, 5936, 6000, 6064, 6128, 6192, 6256, 6320, 6384, 6448, 6512, 6576, 6640, 6704, 6768, 6832, 6896, 6960, 7024, 7088, 7152, 7216, 7280, 7344, 7408, 7472, 7536, 7600, 7664, 7728, 7792, 7856, 7920, 7984, 8048, 8112, 8176, 8240, 8304, 8368, 8432, 8496, 8560, 8624, 8688, 8752, 8816, 8880, 8944, 9008, 9072, 9136, 9200, 9264, 9328, 9392, 9456, 9520, 9584, 9648, 9712, 9776, 9840, 9904, 9968, 10032, 10096, 10160, 10224, 10288, 10352, 10416, 10480, 10544, 10608, 10672, 10736, 10800, 10864, 10928, 10992};
    int length=sizeof(my_list)/sizeof(my_list[0]);
    printf("%d\n",length);
    //set threshold value if given by user
	// if(argc>1){
	// 	thresholdSize = atoi(argv[1]);  
	// }
	
	//vectors storing execution time values
	// vector<double> standardTime(sampleSize);      
	// vector<double> strassenTime(sampleSize);
	
	for (int k = 0; k < length; k++) {
		//initialize vectors for matrices a, b, c: a*b = c
        if(my_list[k]<=3500){
        int matSize = my_list[k];
		vector< vector<double> > 
			a(matSize,vector<double>(matSize)),
			b(matSize,vector<double>(matSize)),
			c(matSize,vector<double>(matSize));	
		initMat(a,b,matSize);
		double l[5];
		// //standard execution		
		// startTime = time(0);
		// multiplyMatStandard(a,b,c,matSize);
		// elapsedTime = time(0) - startTime;
		// standardTime[k] = elapsedTime;
		//multiplication using Strassen'
		if(my_list[k]<=1000){
		for(int j=0;j<5;j++){
        cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        // cout<<startTime<<endl;
		multiplyStrassen(a,b,c,matSize);
        cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
		l[j]=cpu_elapsed_time_ms;
		//printf("%lf\n",cpu_elapsed_time_ms);
	}// elapsedTime = time(0) - startTime;
		// double duration = elapsedTime;
        // clock_t begin1 = clock();
		// multiplyStrassen(a,b,c,matSize);
		// clock_t end1 = clock();
		// double time_spent1 = (double)1000*(end1 - begin1) / CLOCKS_PER_SEC;
		double avg;
	    avg=(l[0]+l[1]+l[2]+l[3]+l[4])/5;
		cout << "Using Milliseconds Clock: (AVG)"<< endl;
		cout << " CPU time taken to execute for strassen matrices of size - " 
		<< matSize << " : " <<avg<<" ms"<< endl;
		cout << endl;
        fprintf(fp,"\n%s,%d,%lf",algoname,matSize,avg);
}else{
	int matSize = my_list[k];
		vector< vector<double> > 
			a(matSize,vector<double>(matSize)),
			b(matSize,vector<double>(matSize)),
			c(matSize,vector<double>(matSize));	
		initMat(a,b,matSize);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	// cout<<startTime<<endl;
	multiplyStrassen(a,b,c,matSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
	cout << "Using Milliseconds Clock:(else above 1000) "<< endl;
		cout << " CPU time taken to execute for strassen matrices of size - " 
		<< matSize << " : " <<cpu_elapsed_time_ms<<" ms"<< endl;
		cout << endl;
        fprintf(fp,"\n%s,%d,%lf",algoname,matSize,cpu_elapsed_time_ms);
}
	// }
	// cout << "Standard multiplication"<< endl;
	// standardMean = calculateMean(standardTime, sampleSize);
	// cout << "Average time taken to execute matrices of size - " 
	// 	<< matSize << " : " << standardMean << endl;
	// cout << endl;
	// cout << "Multiplication using given Original Clock Strassen's"<< endl;
	// cout << "Average time taken to execute matrices of size - " 
	// 	<< matSize << " : " << time_spent1 << endl;
	// cout << endl;
	
	// cout << "Speed up gained by using Strassen's-" << matSize 
	// 	<< " : " << standardMean/strassenMean << endl;
	// cout << endl;
}else{
	        cout<<"MAX LIMIT"<<endl;
            int max_limit=800000;
            fprintf(fp,"\n%s,%d,%d",algoname,my_list[k],max_limit);
			
        }
    }
	return 0;
}

