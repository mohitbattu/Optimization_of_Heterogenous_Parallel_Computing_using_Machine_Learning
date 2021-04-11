nvcc -c -I /home/suejb/software/meterpu_0.81/ -gencode arch=compute_35,code=sm_35 -std=c++11  -DENABLE_NVML  -DENABLE_PTHREAD -DENABLE_PCM -I/home/suejb/software/PCM/VLatest -I/home/suejb/software/ -I . -O3   SPECAccel-OpenCL.cpp -o SPECAccel-OpenCL.o

nvcc SPECAccel-OpenCL.o -o SPECAccel-OpenCL -gencode arch=compute_35,code=sm_35 -std=c++11  -lnvidia-ml  -lpthread -O3 /home/suejb/software/PCM/VLatest/cpucounters.o /home/suejb/software/PCM/VLatest/msr.o /home/suejb/software/PCM/VLatest/pci.o /home/suejb/software/PCM/VLatest/client_bw.o


nvcc -c -I /home/suejb/software/meterpu_0.81/ -gencode arch=compute_35,code=sm_35 -std=c++11 -DENABLE_NVML -DENABLE_PTHREAD -DENABLE_PCM -I/home/suejb/software/PCM/VLatest -I/home/suejb/software/ -I . -O3  SPECAccel-OpenACC.cpp -o SPECAccel-OpenACC.o

nvcc SPECAccel-OpenACC.o -o SPECAccel-OpenACC -gencode arch=compute_35,code=sm_35 -std=c++11  -lnvidia-ml  -lpthread -O3 /home/suejb/software/PCM/VLatest/cpucounters.o /home/suejb/software/PCM/VLatest/msr.o /home/suejb/software/PCM/VLatest/pci.o /home/suejb/software/PCM/VLatest/client_bw.o


nvcc -c -I /home/suejb/software/meterpu_0.81/ -gencode arch=compute_35,code=sm_35 -std=c++11 -DENABLE_NVML -DENABLE_PTHREAD -DENABLE_PCM -I/home/suejb/software/PCM/VLatest -I/home/suejb/software/ -I . -O3  Rodinia-OpenCL.cpp -o Rodinia-OpenCL.o

nvcc Rodinia-OpenCL.o -o Rodinia-OpenCL -gencode arch=compute_35,code=sm_35 -std=c++11  -lnvidia-ml  -lpthread -O3 /home/suejb/software/PCM/VLatest/cpucounters.o /home/suejb/software/PCM/VLatest/msr.o /home/suejb/software/PCM/VLatest/pci.o /home/suejb/software/PCM/VLatest/client_bw.o


nvcc -c -I /home/suejb/software/meterpu_0.81/ -gencode arch=compute_35,code=sm_35 -std=c++11 -DENABLE_NVML -DENABLE_PTHREAD -DENABLE_PCM -I/home/suejb/software/PCM/VLatest -I/home/suejb/software/ -I . -O3  Rodinia-CUDA.cpp -o Rodinia-CUDA.o

nvcc Rodinia-CUDA.o -o Rodinia-CUDA -gencode arch=compute_35,code=sm_35 -std=c++11  -lnvidia-ml  -lpthread -O3 /home/suejb/software/PCM/VLatest/cpucounters.o /home/suejb/software/PCM/VLatest/msr.o /home/suejb/software/PCM/VLatest/pci.o /home/suejb/software/PCM/VLatest/client_bw.o
