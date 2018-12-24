#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include "nppEnhancement.cuh"

struct transformFunction : thrust::unary_function<Npp8u, Npp8u>
{
	Npp8u min, max;

	transformFunction(Npp8u _min, Npp8u _max) {
		min = _min;
		max = _max;
	}

	__host__ __device__
	void operator()(Npp8u pixelValue) {
		pixelValue = (pixelValue - min) * 255 / (max - min);
	}
};

int main()
{
	Npp8u* img_Host;
	int  nWidth, nHeight, nMaxGray;

	// Load image to the host.
	std::cout << "Load PGM file." << std::endl;
	img_Host = LoadPGM("lena_before.pgm", nWidth, nHeight, nMaxGray);
	
	thrust::device_vector<Npp8u> img_Device(img_Host, img_Host + (nWidth * nHeight));
	
	// Finds minimum and maximum in vector
	// Note: It is more efficient than using min_element and max_element seperately.
	thrust::pair<thrust::device_vector<Npp8u>::iterator, thrust::device_vector<Npp8u>::iterator> minmax = thrust::minmax_element(img_Device.begin(), img_Device.end());

	std::cout << int(*(minmax.first)) << " - " << int(*(minmax.second)) << std::endl;

	transformFunction func(*minmax.first, *minmax.second);

	thrust::for_each(img_Device.begin(), img_Device.end(), func);

	//thrust::host_vector<Npp8u> out_Host(img_Device);

    return 0;
}

