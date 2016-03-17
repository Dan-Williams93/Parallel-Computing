
//reduce using local memory to find the minimum value
__kernel void minVal(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {

		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			
			if (scratch[lid] > scratch[lid + i]){
				scratch[lid] = scratch[lid + i];
			}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if (!lid){
		atomic_min(&B[0], scratch[lid]);
	}
}


//reduce using local memory to find the maximum value
__kernel void maxVal(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);	//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			
			if (scratch[lid] < scratch[lid + i]){
				scratch[lid] = scratch[lid + i];
			}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid){
		atomic_max(&B[0], scratch[lid]);
	}
}


//reduce using local memory to find the sum value
__kernel void sum(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	if (!lid){
		atomic_add(&B[0], scratch[lid]);
	}
}


//a very simple histogram implementation
__kernel void hist(__global const int* A, __global int* H) { 
	int id = get_global_id(0);
	int max = 2147483647;

	//check if value is int max if not add to bin
	if (A[id] != max){

		//assumes that H has been initialised to 0
		int bin_index = A[id] + 25;//take value as a bin index
		
		atomic_inc(&H[bin_index]);//serial operation, not very efficient!
	}
}


//a very simple histogram implementation
__kernel void hist2(__global const int* A, __global int* H, __global int* binCount, __global int* range, __global int* minimum) { 
	int id = get_global_id(0);
	int max = 2147483647;
		
	
	//check if value is int max if not add to bin
	if (A[id] != max ){

		int val = A[id] - minimum[0];
		
		//CALCULATED THE BIN INDEX
		int bin_index = ((val * binCount[0]) / range[0]);

		atomic_inc(&H[bin_index]);//serial operation, not very efficient!
	}
}


// value * number of bins / range 

//shift + minimum value to all 