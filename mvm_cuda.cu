#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#define MSIZE 2
#define BLOCKSIZE 256

__global__ void matrix_vector_mul(float *M, float *V, float *output, int cv){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	if(tid<cv){
        for(int i=0;i<cv;i++){
            output[tid]+=M[i*cv]*V[i];
        }
	}
}

void main(){
	float matrix[MSIZE*MSIZE]={1,0,1,0};
	float vector[MSIZE]={1,1};
	float* m_device;
	float* v_device;
	float* o_device;
	cudaMalloc((void**)&m_device,(MSIZE*MSIZE)*sizeof(float));
	cudaMalloc((void**)&v_device,MSIZE*sizeof(float));
	cudaMalloc((void**)&o_device,MSIZE*sizeof(float));
	cudaMemcpy(m_device, matrix, MSIZE*MSIZE*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(v_device, vector, MSIZE*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemset(o_device,0,MSIZE*sizeof(float));
	
	dim3 dimBlock(BLOCKSIZE);
	int blocknum=(MSIZE+BLOCKSIZE-1)/BLOCKSIZE;
	if(blocknum>65536){
	   int k=(blocknum+65536-1)/65536;
	   dim3 dimGrid(k,65536);
	}else
       dim3 dimGrid(1,(MSIZE+BLOCKSIZE-1)/BLOCKSIZE);

    matrix_vector_mul<<<dimGrid,dimBlock>>>(m_device,v_device,o_device,MSIZE);
    
    float output[MSIZE];
    cudaMemcpy(output,o_device, MSIZE*sizeof(float),cudaMemcpyDeviceToHost);

}