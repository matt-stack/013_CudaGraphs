# 013_CudaGraphs

In this homework we will look at two different codes, both using Cuda Graphs. These codes consist of small kernels and could see some benefit from managing the launch latency overhead better. The two codes are axpy_stream_capture and axpy_cublas, each having a verison with_fixme and from_scratch. We recommend starting with the with_fixme versions of the two code, and then trying the from_scratch if you want a challenge. The with_fixme versions will have spots where you will need to fix to get the code to run, but the framewrok is set in place. The from_scratch versions require you to implement the graph set up and logic by hand.

You can refer to the solutions in the Solutions directory for help/hints when stuck.

### Task 1
#### Stream Capture
This task will be an example of how to use stream capture with Cuda Graphs. We will be creating a graph from a sequence of kernel launchs across two streams.

We will be looking to implement the following graph, which can be helpful to see visually

![](graph_stream_capture.png)

This is the same example from the slides, feel free to refer to them for help and hints.

Go ahead and take a look at the code now to get a sense of the new Graph API calls. On first pass, ignore the Graph APIs and try get a sense of the underlying code and what it is doing. The kernels themselves are not doing any specific math, but simply represent a random small kernel. Remember to think about the function of the two streams and refer back to the picture here to make sure you see the inherient dependencies created by the Cuda EventWait and Signal. 

`bool graphCreated=false;` will be our method doing the set up for the graph only on the first pass (for loop iteration 0), then go straight to launching the graph in each subsequent iteration (1 - (N-1). 

An important distinction is the difference between the type `cudaGraph_t` and `cudaGraphExec_t`. `cudaGraph_t` is used to define the shape and the arguments of the overall graph and `cudaGraphExec_t` is a callable instance of the graph, which has gone through the instantiate step. 


FIXMEs
1. cudaGraphCreate(FIXME, 0);
2. cudaGraphInstantiate(FIXME, graph, NULL, NULL, 0);
3. graphCreated = FIXME;
4. cudaGraphLaunch(FIXME, streams[0]);


### Task 2
#### Explicit Graph Creation w/ Library Call
In this task, we will look at a few of the explicit graph creation API and how to capture a library call with stream capture. A key to this example is remembering while we are using both explicit graph creation and stream capture, both are just ways of defining to a `cudaGraph_t` which we then instantiate into a `cudaGraphExec_t`. 

https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html

This is the documentaion of the current Cuda toolkit graph management API. You can complete this example without consulting the docs by using the slides and context clues in the code, or but taking a look at the definition of `cudaGraphAddChildGraphNode` may help you if you are stuck with the FIXME.

Unlike the first example, 

![](graph_with_library_call.png)

FIXME
1. cudaGraphCreate(FIXME, 0);
2. cudaGraphAddChildGraphNode(FIXME, graph, FIXME, nodeDependencies.size(), libraryGraph);
3. cudaGraphLaunch(FIXME, stream1);

