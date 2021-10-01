#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define N 500000

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

__global__
void saxpy(int n, float a, float * x, float * y)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = a*x[idx] + y[idx];
}

__global__
void kernel_a(float* x, float* y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] += 1;

}

__global__
void kernel_c(float* x, float* y){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < N) y[idx] -= 1;

}

void initialize(float* h_temp){
  for (int i = 0; i < N; ++i){
    h_temp[i] = (float)i;
  }
}

int main(){

cudaStream_t stream1;

cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

cublasHandle_t cublas_handle;
cublasCreate(&cublas_handle);

float* h_x;
float* h_y;

h_x = (float*) malloc(N * sizeof(float));
h_y = (float*) malloc(N * sizeof(float));

printf("yes?\n");
initialize(h_x);
initialize(h_y);
printf("yes?\n");

float* d_x;
float* d_y;
float d_a = 5.0;

cudaMalloc((void**) &d_x, N * sizeof(float));
cudaMalloc((void**) &d_y, N * sizeof(float));

printf("yes?\n");
cublasSetVector(N, sizeof(h_x[0]), h_x, 1, d_x, 1); // similar to cudaMemcpy
cublasSetVector(N, sizeof(h_y[0]), h_y, 1, d_y, 1); // similar to cudaMemcpy
cudaCheckErrors("Mallocing failed");

cudaGraph_t graph; // main graph
cudaGraph_t libraryGraph; // sub graph for cuBLAS call
printf("yes?\n");
std::vector<cudaGraphNode_t> nodeDependencies;
//cudaGraphNode_t nodeDependencies[];
cudaGraphNode_t kernelNode1, kernelNode2, libraryNode;

cudaKernelNodeParams kernelNode1Params {0};
cudaKernelNodeParams kernelNode2Params {0};

void *kernelArgs[2] = {(void *)&d_x, (void *)&d_y};

int threads = 512;
int blocks = (N + (threads - 1) / threads);

kernelNode1Params.func = (void *)kernel_a;
//kernelNode1Params.gridDim = dim3(blocks, 1, 1);
//kernelNode1Params.blockDim = dim3(threads, 1, 1);
kernelNode1Params.gridDim = dim3(1024, 1, 1);
kernelNode1Params.blockDim = dim3(512, 1, 1);
kernelNode1Params.sharedMemBytes = 0;
kernelNode1Params.kernelParams = (void **)kernelArgs;
kernelNode1Params.extra = NULL;
printf("yes!?\n");

//cudaGraphAddKernelNode(&kernelNode1, graph, nodeDependencies.data(),
cudaGraphAddKernelNode(&kernelNode1, graph, NULL,
                         0, &kernelNode1Params);
printf("yes!\n");
cudaCheckErrors("Adding kernelNode1 failed");

nodeDependencies.push_back(kernelNode1);

cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

cublasSaxpy(cublas_handle, N, &d_a, d_x, 1, d_y, 1);

cudaStreamEndCapture(stream1, &libraryGraph);

cudaGraphAddChildGraphNode(&libraryNode, graph, nodeDependencies.data(),
                             nodeDependencies.size(), libraryGraph);
cudaCheckErrors("Adding libraryNode failed");

nodeDependencies.clear();
nodeDependencies.push_back(libraryNode);

kernelNode2Params.func = (void *)kernel_c;
kernelNode2Params.gridDim = dim3(blocks, 1, 1);
kernelNode2Params.blockDim = dim3(threads, 1, 1);
kernelNode2Params.sharedMemBytes = 0;
kernelNode2Params.kernelParams = (void **)kernelArgs;
kernelNode2Params.extra = NULL;

cudaGraphAddKernelNode(&kernelNode2, graph, nodeDependencies.data(),
                         nodeDependencies.size(), &kernelNode2Params);
cudaCheckErrors("Adding kernelNode1 failed");

nodeDependencies.clear();
nodeDependencies.push_back(kernelNode2);

cudaGraphNode_t *nodes = NULL;
size_t numNodes = 0;
cudaGraphGetNodes(graph, nodes, &numNodes);
cudaCheckErrors("Graph instantiation failed");
printf("Number of the nodes in the graph = %zu\n", numNodes);

cudaGraphExec_t instance;
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
cudaCheckErrors("Graph instantiation failed");

for (int i = 0; i < 100; ++i){
// Launch the graph instance
//printf("launching graph\n");
cudaGraphLaunch(instance, stream1);
cudaStreamSynchronize(stream1);

}

cudaMemcpy(h_y, d_y, N, cudaMemcpyDeviceToHost);

cudaDeviceSynchronize();

for (int i = 0; i < N; ++i){
//    printf("%2.0f ", h_y[i]);
}
printf("\n");






}
