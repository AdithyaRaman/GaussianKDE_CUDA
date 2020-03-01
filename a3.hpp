/*  ADITHYA
 *  RAMAN
 *  araman5
 */

#ifndef A3_HPP
#define A3_HPP


#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>

#define PI 3.14
#define BLOCK_SIZE 1024

__device__ float K(float val)
{
    return expf(-val*val/2)/sqrtf(2*PI);
}

float K2(float val)
{
    return expf(-val*val/2)/sqrtf(2*PI);
}


bool cmpf(float A, float B, float epsilon = 0.0000000001f)
{   
 
    return (fabs(A - B) < epsilon);
}

__global__ void computeKernalFunction(int n, float h,int s ,float *x, float *y)
{

    __shared__ float buf[BLOCK_SIZE];

    int gidx = threadIdx.x +s;
    int lidx = threadIdx.x;
    buf[lidx] = x[gidx];

    float x_val = x[gidx], sum=0;
    __syncthreads();
    if (gidx < n)
    {
        for (int i = 0; i < BLOCK_SIZE; i++)
        {
            sum += K((x_val - buf[i])/h);
        }
        y[gidx] = sum/(n*h);
    }
}



void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {
    float *input, *output;

    long size = x.size() * sizeof(float);
    long block_num = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void **)&input, size);
    cudaMalloc((void **)&output, size);

    cudaMemcpy(input, x.data(), size, cudaMemcpyHostToDevice);

    /*Call kernel here*/

    for(int i = 0; i< block_num;i++)
    {   
        computeKernalFunction<<<block_num, BLOCK_SIZE>>>(n, h,i*BLOCK_SIZE, input, output);
    }
  
    /*Retrieve the values*/
    cudaMemcpy(y.data(), output, size, cudaMemcpyDeviceToHost);

  
    /*Clean up memory */
    cudaFree(input);
    cudaFree(output);


    /****************************************BLOCK OF CODE TO VERIFY CORRECTNESS *****************************************************************/
    /****************************************UNCOMMENT AND CHECK FOR RELATIVELY SMALL VALUE OF N *************************************************/
    // std::cout<<"Cuda Output:" << std::endl;
    // for (int i = 0; i < y.size(); i++)
    //     std::cout << y[i] << ' ';
    // std::cout << std::endl;

    // std::cout<<"Sequential Output:" << std::endl;
    // std::vector<float> y2(n, 0.0);

    // float sum =0.0;
    // for(int i=0; i<n; i++)
    // {
    //     sum =0.0;
    //     for(int j=0; j<n; j++)
    //     {
    //         sum+= K2( (x[i] - x[j]) /h);
    //     }
    //     y2[i] = sum/(n*h);
    // }
    // bool flag = true;
    // for (int i = 0; i < y2.size(); i++)
    // {
    //     std::cout << y2[i] << ' ';
    //     if(cmpf(y[i],y2[i]))
    //     {
    //         flag = false;
    //         //std::cout<<"\nSomething wrong at:"<<i<<"--"<<y[i]<<" "<<y2[i]<<std::endl;
    //         //break;
    //     }
    // }
    // std::cout << std::endl;

    // if(flag)
    // {
    //     std::cout<<"Everything seems to be working"<<std::endl;
    // }
    // else
    // {
    //     std::cout<<"Oops, Parallel output not equal sequential output. Something is wrong"<<std::endl;
    // }
    

} // gaussian_kde


#endif // A3_HPP
