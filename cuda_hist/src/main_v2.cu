

#include <chrono>
#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

const size_t kYFactor = 8;

__global__ void subhist(
    uint8_t *image,
    size_t w,
    size_t h,
    size_t memPitch,
    uint32_t *subHistogram,
    size_t nbins,
    size_t nsubhist,
    size_t subHistogramPitch)
{
    extern __shared__ uint32_t localHist[];

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idy = blockIdx.y * blockDim.y + threadIdx.y;

    uint32_t xstride = gridDim.x*blockDim.x;
    uint32_t ystride = gridDim.y*blockDim.y*kYFactor;

    uint32_t sid = threadIdx.x;
    for(int i=sid; i<nbins; i+=blockDim.x)
    {
        localHist[i] = 0;
    }
    __syncthreads();

    for(int j=idx; j<w; j+=xstride)
    {
        for(int i=idy*kYFactor; i<h; i+=ystride)
        {
            for(int k=0; k<min(kYFactor, h-i); k++)
            {
                uint8_t pv = image[(i+k)*memPitch + j];
                atomicAdd(&localHist[pv], 1);
            }
        }
    }
    __syncthreads();

    uint32_t subHistId = idy * gridDim.x + blockIdx.x;
    for(int i=sid; i<nbins; i+=blockDim.x)
    {
        subHistogram[subHistId*subHistogramPitch + i] = localHist[i];
    }
}

__global__ void sumSubHistograms(
    uint32_t *subHistogram,
    size_t nbins,
    size_t nsubhist,
    size_t subHistogramPitch,
    uint32_t *histogram)
{
    uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t step = blockDim.x * gridDim.x;

    for(int i=sid; i<nbins; i+=step)
    {
        uint32_t sum = 0;
        for(int j=0; j<nsubhist; j++)
        {
            sum += subHistogram[j*subHistogramPitch + i];
        }
        histogram[i] = sum;
    }
}

int main(int argc, char** argv)
{
    int device;
    cudaDeviceProp properties = {};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    cout << "device: " << device << endl;
    cout << "        multiProcessorCount: " << properties.multiProcessorCount << endl;
    cout << "maxThreadsPerMultiProcessor: " << properties.maxThreadsPerMultiProcessor << endl;

    chrono::time_point<chrono::steady_clock> t1, t2;

    cv::Mat image;
    cv::Mat image1;
    cv::Mat cvHist;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    image1 = cv::Mat::zeros(image.rows, image.cols, image.depth());

    float range[] = { 0, 256 };
    const float* histRange = { range };
    int channels = 0;
    int hs = 256;

    t1 = chrono::steady_clock::now();
    cv::calcHist(&image, 1, &channels, cv::Mat(), cvHist, 1, &hs, &histRange, true, false);
    t2 = chrono::steady_clock::now();
    uint64_t duration = chrono::duration_cast<chrono::nanoseconds>(t2-t1).count();
    cout << "CPU compute time: " << duration / 1000.f << "us" << endl << endl;

    cv::namedWindow( "Display window" );
    cv::imshow( "Display window", image );
    cv::waitKey(0);

    if(image.depth()!=CV_8U)
    {
        cout << "incompatible data type" << endl;
        return 0;
    }

    size_t nbins = 256;
    size_t width = image.cols;
    size_t height = image.rows;
    size_t npix = width * height;

    uint8_t *inputDevice;
    uint8_t *outputDevice;

    uint32_t *histogram;
    uint32_t *histogramHost;
    uint32_t *subHistogram;

    size_t inputPitch;
    size_t outputPitch;
    size_t subHistogramPitch;

    size_t bsx = 512;
    size_t gsx = 1;
    size_t gsy = properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor / bsx;
    size_t smem = sizeof(uint32_t) * nbins;
    size_t nsubhist = gsx * gsy;
    dim3 subHistGridDim(gsx, gsy, 1);
    dim3 subHistBlockDim(bsx, 1, 1);
    dim3 sumGridDim((nbins+bsx-1)/bsx, 1, 1);
    dim3 sumBlockDim(bsx, 1, 1);

    cout << "image width: " << width << endl;
    cout << "image height: " << height << endl;

    CUDA_CHECK(cudaMallocPitch(&inputDevice, &inputPitch, width, height));
    CUDA_CHECK(cudaMallocPitch(&outputDevice, &outputPitch, width, height));
    CUDA_CHECK(cudaMallocPitch(&subHistogram, &subHistogramPitch, sizeof(uint32_t)*nbins, nsubhist));
    CUDA_CHECK(cudaMallocHost(&histogramHost, sizeof(uint32_t) * nbins));
    CUDA_CHECK(cudaMalloc(&histogram, sizeof(uint32_t) * nbins));
    CUDA_CHECK(cudaHostRegister(image.data, sizeof(uint8_t)*npix, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(image1.data, sizeof(uint8_t)*npix, cudaHostRegisterDefault));

    CUDA_CHECK(cudaMemcpy2D(inputDevice, inputPitch, image.data, width, width, height, cudaMemcpyHostToDevice));

    double durationSum = 0.0;
    for(int i=0; i<100; i++)
    {
        t1 = chrono::steady_clock::now();
        subhist<<<subHistGridDim, subHistBlockDim, smem>>>(inputDevice, width, height, inputPitch, subHistogram, nbins, nsubhist, subHistogramPitch);
        sumSubHistograms<<<sumGridDim, sumBlockDim>>>(subHistogram, nbins, nsubhist, subHistogramPitch, histogram);
        CUDA_CHECK(cudaDeviceSynchronize());
        t2 = chrono::steady_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
        // cout << "GPU duration: " << duration / 1000.f << "us" << endl;
        durationSum += duration;
    }
    cout << "Average GPU compute time (100 runs): " << durationSum / 100.f << "us" << endl << endl;


    CUDA_CHECK(cudaMemcpy2D(image1.data, width, inputDevice, inputPitch, width, height, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(histogramHost, histogram, sizeof(uint32_t) * nbins, cudaMemcpyDeviceToHost));
    for(size_t i=0; i<nbins; i++)
    {
        if(histogramHost[i] != (uint32_t)cvHist.at<float>(i))
        {
            cout << "cuda histogram is different from OpenCV histogram!!" << endl;
            cout << i << ": " << histogramHost[i] <<endl;
        }
    }

    CUDA_CHECK(cudaFree(inputDevice));
    CUDA_CHECK(cudaFree(outputDevice));
    CUDA_CHECK(cudaFree(subHistogram));
    CUDA_CHECK(cudaFreeHost(histogramHost));
    CUDA_CHECK(cudaFree(histogram));
    CUDA_CHECK(cudaHostUnregister(image.data));
    CUDA_CHECK(cudaHostUnregister(image1.data));

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
