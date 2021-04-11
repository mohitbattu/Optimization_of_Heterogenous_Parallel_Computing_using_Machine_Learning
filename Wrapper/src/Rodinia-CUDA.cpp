#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>


#include <MeterPU.h>
using namespace std;

#ifdef WIN32
FILE *popen ( const char* command, const char* flags) {return _popen(command,flags);}
int pclose ( FILE* fd) { return _pclose(fd);}
#endif

bool checkExists (const string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

void writeToCSV(string file_name, string application, float time, float gpu_energy, float cpu_energy, string file_size, string tune, int iteration){

	ofstream myfile;
	string header = "";
	if(checkExists(file_name+".csv") == false)
    {
    	cout<<"file does not exist. Create header. \n";
        header = "#,application,tune,size,time(s),CPU_Energy(milliJoules),GPU_Energy(milliJoules)\n";
    }

	myfile.open (file_name+".csv", std::ios_base::app);

	if ( header != "" )
	{
	   myfile << header;
	}
	if(time == 0) {
        myfile << "Average, , , , , , \n"; //put an empty line between each application
        myfile << ", , , , , , \n"; //put an empty line between each application
    }
	else
		myfile << iteration << "," << application << "," << tune << "," << file_size << "," << time << "," << cpu_energy << "," << gpu_energy << "\n";

	myfile.close();
}

int main(int argc, char* argv[])
{
	using namespace MeterPU;


	const int iterations = 20;
	const string base_dir = "/home/suejb/benchmarks/Rodinia/rodinia_3.1/cuda/";
	const string file_name = "rodinia_cuda_ida";
	const string file_size = "ref";
	const string tune = "base";

	const string applications[] = {"b+tree", "backprop", "bfs", "cfd", "heartwall", "hotspot", "hotspot3D", "kmeans", "lavaMD", "lud", "myocyte", "nw", "particlefilter", "srad/srad_v1", "streamcluster"};

	for (int i = 0; i < 15; i++) {
		for(int j = 0; j< iterations; j++) {
			string command = "cd "+base_dir+applications[i]+ "; ./run";

			float time = 0;

			const clock_t begin_time = clock();


			//Initialize a meter with GPU energy of default device id 0
			Meter<NVML_Energy<>> gpu_energy;
            Meter<PCM_Energy> cpu_energy;

		    char psBuffer[128];
		    FILE *iopipe;

		 	gpu_energy.start();
		 	cpu_energy.start();

		    if( (iopipe = popen( command.c_str(), "r" )) == NULL )
		        exit( 1 );

		    while( !feof( iopipe ) )
		    {
		        if( fgets( psBuffer, 128, iopipe ) != NULL )
		            printf( psBuffer );
		    }

		    gpu_energy.stop();

			gpu_energy.calc();

            cpu_energy.stop();
            cpu_energy.calc();

			time = float( clock () - begin_time ) /  CLOCKS_PER_SEC;

			cout<<"GPU Energy consumed is: "<<gpu_energy.get_value()<<" milliJ."<<endl;
			cout<<"CPU Energy consumed is: "<<cpu_energy.get_value()<<" milliJ."<<endl;
			// gpu_energy.show_meter_reading();

			writeToCSV(file_name, applications[i], time, gpu_energy.get_value(), cpu_energy.get_value(), file_size, tune, j);
			printf( "\nProcess returned %d\n", pclose( iopipe ) );
		}
		writeToCSV(file_name, "", 0, 0, 0, "", "", 0); // write an empty line
	}


    return 0;
}
