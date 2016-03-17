#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	////Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	ifstream inFile; //create obkject of class
	inFile.open("temp_lincolnshire.txt"); //open file

	//check for error
	if (inFile.fail()) {
		cerr << "Error opening File" <<endl;
		exit(1);
	}
	else {
		cout << "Reading file, this may take some time" << endl;
	}

	int count = 0;
	string location;
	int year;
	int month;
	int day;
	int time;
	float temp;

	std::vector<string> locations;
	std::vector<int> years;
	std::vector<int> months;
	std::vector<int> days;
	std::vector<int> times;
	std::vector<int> temps;

	//read file until end is reached
	while (!inFile.eof()) {
		inFile >> location >> year >> month >> day >> time >> temp;

		locations.push_back(location);
		years.push_back(year);
		months.push_back(month);
		days.push_back(day);
		times.push_back(time);
		temps.push_back((int)temp);
		count++;
	}

	cout << "File read complete, " << count << " items found" << endl;

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//input
		std::vector<int> A = temps; //AVG / SUM
		
		std::vector<int> minTemps = temps; //MIN
		
		std::vector<int> maxTemps = temps; //MAX


		//number of input elements
		size_t original_Length = A.size();

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 1024;
		
		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements so that the total will not be affected
		if (padding_size) {
			
			//AVERAGE
			std::vector<int> A_ext(local_size-padding_size, 0);

			//MINIMUM
			std::vector<int> minTemps_ext(local_size - padding_size, INT_MAX);

			//MAXIMUM
			std::vector<int> maxTemps_ext(local_size - padding_size, INT_MIN);
			
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end()); 
			minTemps.insert(minTemps.end(), minTemps_ext.begin(), minTemps_ext.end());
			maxTemps.insert(maxTemps.end(), maxTemps_ext.begin(), maxTemps_ext.end());
		}

		size_t input_elements = A.size();	//number of input elements with padding
		size_t input_size = A.size()*sizeof(mytype);	//size in bytes
		size_t nr_groups = input_elements / local_size;

		//host - output
		//max - min + 1 for hist size
		std::vector<mytype> output(1);
		size_t output_size = output.size()*sizeof(mytype);//size in bytes

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_minTemps(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_maxTemps(context, CL_MEM_READ_ONLY, input_size);

		cl::Buffer buffer_output(context, CL_MEM_READ_WRITE, output_size);

		//-----------------------------------------------------------------------------------------------------------------

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_minTemps, CL_TRUE, 0, input_size, &minTemps[0]);
		queue.enqueueFillBuffer(buffer_output, INT_MAX, 0, output_size);

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_1 = cl::Kernel(program, "minVal");
		kernel_1.setArg(0, buffer_minTemps);
		kernel_1.setArg(1, buffer_output);
		kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output_size, &output[0]);

		int minimumTemp = output[0];

		//------------------------------------------------------------------------------------------------------------------

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_maxTemps, CL_TRUE, 0, input_size, &maxTemps[0]);
		queue.enqueueFillBuffer(buffer_output, INT_MIN, 0, output_size);

																   //5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_2 = cl::Kernel(program, "maxVal");
		kernel_2.setArg(0, buffer_maxTemps);
		kernel_2.setArg(1, buffer_output);
		kernel_2.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

																 //call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_2, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output_size, &output[0]);

		int maximumTemp = output[0];

		//-------------------------------------------------------------------------------------------------------------------

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_output, 0, 0, output_size);

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_3 = cl::Kernel(program, "sum");
		kernel_3.setArg(0, buffer_A);
		kernel_3.setArg(1, buffer_output);
		kernel_3.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

																 //call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_3, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_output, CL_TRUE, 0, output_size, &output[0]);

		double avgTemp = 1.00* output[0] / original_Length;

		//-----------------------------------------------------------------------------------------------------------------

		int binCount = 20;
		int range = (maximumTemp - minimumTemp) + 1;
		int minVal = minimumTemp;

		cout << "How many bins would you like in your histogram" << endl;
		cin >> binCount;
				
		std::vector<mytype> histOutput(binCount);
		size_t hist_output_size = histOutput.size()*sizeof(mytype);//size in bytes

		std::vector<mytype> his_bin = { binCount };
		size_t hist_binCount_size = his_bin.size()*sizeof(mytype);//size in bytes

		std::vector<mytype> his_range = { range };
		size_t hist_range_size = his_range.size()*sizeof(mytype);//size in bytes

		std::vector<mytype> his_min = { minimumTemp };
		size_t hist_min_size = his_min.size()*sizeof(mytype);//size in bytes
		
		//device - buffer
		cl::Buffer buffer_histOutput(context, CL_MEM_READ_WRITE, hist_output_size);
		cl::Buffer buffer_bin(context, CL_MEM_READ_WRITE, hist_binCount_size); 
		cl::Buffer buffer_range(context, CL_MEM_READ_WRITE, hist_range_size); 
		cl::Buffer buffer_minimum(context, CL_MEM_READ_WRITE, hist_min_size); 

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_minTemps, CL_TRUE, 0, input_size, &minTemps[0]);
		queue.enqueueFillBuffer(buffer_histOutput, 0, 0, hist_output_size);
		queue.enqueueWriteBuffer(buffer_bin, CL_TRUE, 0, hist_binCount_size, &his_bin[0]); 
		queue.enqueueWriteBuffer(buffer_range, CL_TRUE, 0, hist_range_size, &his_range[0]);
		queue.enqueueWriteBuffer(buffer_minimum, CL_TRUE, 0, hist_min_size, &his_min[0]);

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_4 = cl::Kernel(program, "hist2");
		kernel_4.setArg(0, buffer_minTemps);
		kernel_4.setArg(1, buffer_histOutput);
		kernel_4.setArg(2, buffer_bin);
		kernel_4.setArg(3, buffer_range);
		kernel_4.setArg(4, buffer_minimum);
		
		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(kernel_4, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_histOutput, CL_TRUE, 0, hist_output_size, &histOutput[0]);


		//--------------------------------------------------------------------------------------------------------------------

		std::cout << "\nMinimum Temp = " << minimumTemp << std::endl;
		std::cout << "Maximum Temp = " << maximumTemp << std::endl;
		std::cout << "Average Temp = " << avgTemp << std::endl;


		std::cout << "\nHistogram:\n " << std::endl;
		std::cout << histOutput << std::endl;

		cin.get();

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}	


	cin.get();

	return 0;
}
