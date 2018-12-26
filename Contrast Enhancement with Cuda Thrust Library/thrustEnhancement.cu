#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include "nppEnhancement.cuh"

struct transformFunction 
{
	Npp8u min, max;

	transformFunction(Npp8u _min, Npp8u _max) {
		min = _min;
		max = _max;
	}

	// The function applied on GPU.
	// Reference operator is important. Otherwise we need another memory space to copy the result. Now we apply changes on input.
	__device__ void operator()(Npp8u &pixelValue) {
		pixelValue = (pixelValue - min) * 255 / (max - min);
	}
};

int main()
{
	// Runs contrast enhancement with Thrush or NPP implementation.
	// Both implementation measures its running time.
	bool runWithThrust = true;

	if (runWithThrust) {

		Npp8u* img_Host;
		int  nWidth, nHeight, nMaxGray;

		// Load image to the host.
		std::cout << "Load PGM file." << std::endl;
		img_Host = LoadPGM("lena_before.pgm", nWidth, nHeight, nMaxGray);

		// Performance clock start (just before using GPU)
		auto start = std::chrono::steady_clock::now();

		// Load image onto GPU
		thrust::device_vector<Npp8u> img_Device(img_Host, img_Host + (nWidth * nHeight));

		// Finds minimum and maximum in vector
		// Note: It is more efficient than using min_element and max_element seperately.
		thrust::pair<thrust::device_vector<Npp8u>::iterator, thrust::device_vector<Npp8u>::iterator> minmax = thrust::minmax_element(img_Device.begin(), img_Device.end());

		// Transform each element of image by the operator() in transformFunction.
		// It basically applys contrast enhancement formula.
		thrust::for_each(img_Device.begin(), img_Device.end(), transformFunction(*minmax.first, *minmax.second));

		// Copy transformed image into host vector.
		// Then convert vector into raw host pointer.
		thrust::host_vector<Npp8u> out_Host(img_Device);
		Npp8u* out_ptr_Host = out_Host.data();

		// Performance clock stop (just after getting data from GPU)
		auto end = std::chrono::steady_clock::now();
		auto diff = end - start;
		auto diff_sec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
		std::cout << "Computation time(ms): " << diff_sec.count() << std::endl;

		// Write result.
		WritePGM("lena_after_thrust.pgm", out_ptr_Host, nWidth, nHeight, nMaxGray);
	}
	else {
		nppEnhancement();
	}

    return 0;
}

