#include <stdio.h>

#include <cuda_runtime_api.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define M 1024
#define N 500000

__global__
void saxpy(int n, float a, float * x, float * y)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = a*x[idx] + y[idx];
}

__global__
void kernel_a(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_b(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_c(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

__global__
void kernel_d(float * x, float * y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] = 2.0*x[idx] + y[idx];

}

int main(){

cudaEvent_t event1;
cudaEvent_t event2;

cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);

const int num_streams = 2;

cudaStream_t streams[num_streams];

for (int i = 0; i < num_streams; ++i){
cudaStreamCreate(&streams[i]);
}

cudaError_t cuda_error;

float* h_x;
float* h_y;

h_x = (float*) malloc(N * sizeof(float));

for (int i = 0; i < N; ++i){
    h_x[i] = (float)i;
//    printf("%2.0f ", h_x[i]);
}
printf("\n");

h_y = (float*) malloc(N * sizeof(float));

for (int i = 0; i < N; ++i){
    h_y[i] = (float)i;
//    printf("%2.0f ", h_y[i]);
}
printf("\n");

float* d_x;
float* d_y;

cudaMalloc((void**) &d_x, N * sizeof(float));
cudaMalloc((void**) &d_y, N * sizeof(float));

cudaMemcpy(d_x, h_x, N, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, N, cudaMemcpyHostToDevice);

bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

checkCudaErrors(cudaGraphCreate(&graph, 0));

for (int i = 0; i < 100; ++i){
if (graphCreated == false){
// Starting stream capture
cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeGlobal);

kernel_a<<<1024, 512, 0, streams[0]>>>(d_x, d_y);

cudaEventRecord(event1, streams[0]);

kernel_b<<<1024, 512, 0, streams[0]>>>(d_x, d_y);

cudaStreamWaitEvent(streams[1], event1);

kernel_c<<<1024, 512, 0, streams[1]>>>(d_x, d_y);

cudaEventRecord(event2, streams[1]);

cudaStreamWaitEvent(streams[0], event2);

kernel_d<<<1024, 512, 0, streams[0]>>>(d_x, d_y);

cudaStreamEndCapture(streams[0], &graph);

// Creating the graph instance

cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

graphCreated = true;
}
// Launch the graph instance
//printf("launching graph\n");
cudaGraphLaunch(instance, streams[0]);
cudaStreamSynchronize(streams[0]);

}

cudaMemcpy(h_y, d_y, N, cudaMemcpyDeviceToHost);

cudaDeviceSynchronize();

for (int i = 0; i < N; ++i){
//    printf("%2.0f ", h_y[i]);
}
printf("\n");






}
