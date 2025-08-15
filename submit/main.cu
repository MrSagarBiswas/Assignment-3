#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <climits>
using namespace std;

#define MOD 1000000007

// Structure to represent an edge
struct Edge {
    int src, dest, weight;
    char type;
    
    // Function to determine effective weight based on edge type
    __host__ __device__ int effectiveWeight() const {
        if (type == 'g') return 2 * weight;
        if (type == 't') return 5 * weight;
        if (type == 'd') return 3 * weight;
        return weight;
    }
};

// CUDA device function to find the root of a set with path compression
__device__ int d_find(int *parent, int i) {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]]; // Path compression
        i = parent[i];
    }
    return i;
}

// CUDA kernel to reset cheapest array
__global__ void resetCheapest(long long *cheapest, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        cheapest[i] = LLONG_MAX;
    }
}

// CUDA kernel to find the cheapest edge for each component
__global__ void findCheapestEdge(Edge *edges, int E, int *parent, long long *cheapest) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        Edge e = edges[idx];
        int uComp = d_find(parent, e.src);
        int vComp = d_find(parent, e.dest);
        if (uComp == vComp) return;
        long long candidate = (long long)e.effectiveWeight() * E + idx;
        atomicMin(&cheapest[uComp], candidate);
        atomicMin(&cheapest[vComp], candidate);
    }
}

// CUDA kernel to perform unions and path compression
__global__ void performUnionsAndCompress(Edge *edges, int E, int V, int *parent, long long *cheapest, long long *d_cost, int *edgeUsed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        long long candidate = cheapest[i];
        if (candidate != LLONG_MAX) {
            int edgeIdx = candidate % E;
            Edge e = edges[edgeIdx];
            int pu = d_find(parent, e.src);
            int pv = d_find(parent, e.dest);
            if (pu != pv) {
                int old = atomicCAS(&parent[pv], pv, pu);
                if (old == pv) {
                    if (atomicCAS(&edgeUsed[edgeIdx], 0, 1) == 0) {
                        atomicAdd((unsigned long long*)d_cost, (unsigned long long)e.effectiveWeight());
                    }
                }
            }
        }
    }
}

// CUDA kernel to compress paths
__global__ void compressPaths(int *parent, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        d_find(parent, i);
    }
}

// CUDA kernel to count the number of components
__global__ void countComponents(int *parent, int V, unsigned int *d_numComponents, long long *d_cost) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    unsigned int local_count = 0;
    for (unsigned int i = tid; i < V; i += stride) {
        if (d_find(parent, i) == i) {
            local_count++;
        }
    }
    atomicAdd(d_numComponents, local_count);
    if(*d_numComponents==1) *d_cost %= MOD;
}

int main() {
    int V, E;
    cin >> V >> E;
    vector<Edge> edges(E);
    
    // Read input edges
    for (int i = 0; i < E; i++) {
        int u, v, wt;
        string s;
        cin >> u >> v >> wt >> s;
        edges[i] = {u, v, wt, s[0]};
    }
    
    // Allocate memory on GPU
    Edge *d_edges;
    int *d_parent, *d_edgeUsed;
    long long *d_cheapest, *d_cost;
    unsigned int *d_numComponents;
    
    cudaMalloc(&d_edges, E * sizeof(Edge));
    cudaMalloc(&d_parent, V * sizeof(int));
    cudaMalloc(&d_cheapest, V * sizeof(long long));
    cudaMalloc(&d_cost, sizeof(long long));
    cudaMalloc(&d_edgeUsed, E * sizeof(int));
    cudaMalloc(&d_numComponents, sizeof(unsigned int));
    
    // Copy edges to GPU memory
    cudaMemcpy(d_edges, edges.data(), E * sizeof(Edge), cudaMemcpyHostToDevice);
    
    // Initialize parent array
    vector<int> h_parent(V);
    for (int i = 0; i < V; i++) h_parent[i] = i;
    cudaMemcpy(d_parent, h_parent.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize cost and edge usage
    long long mst_cost = 0;
    cudaMemcpy(d_cost, &mst_cost, sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemset(d_edgeUsed, 0, E * sizeof(int));
    
    // Define grid and block sizes
    int threadsPerBlock = 512;
    int vertexBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
    int edgeBlocks = (E + threadsPerBlock - 1) / threadsPerBlock;
    
    unsigned int numComponents = V;
    
    auto start = chrono::high_resolution_clock::now();
    
    // Boruvka's algorithm loop
    while (numComponents > 1) {
        resetCheapest<<<vertexBlocks, threadsPerBlock>>>(d_cheapest, V);
        findCheapestEdge<<<edgeBlocks, threadsPerBlock>>>(d_edges, E, d_parent, d_cheapest);
        performUnionsAndCompress<<<vertexBlocks, threadsPerBlock>>>(d_edges, E, V, d_parent, d_cheapest, d_cost, d_edgeUsed);
        compressPaths<<<vertexBlocks, threadsPerBlock>>>(d_parent, V);
        
        cudaMemset(d_numComponents, 0, sizeof(unsigned int));
        countComponents<<<4096, threadsPerBlock>>>(d_parent, V, d_numComponents, d_cost);
        cudaMemcpy(&numComponents, d_numComponents, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }
    
    // Retrieve final MST cost
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    cudaMemcpy(&mst_cost, d_cost, sizeof(long long), cudaMemcpyDeviceToHost);
    
    // Output results
    ofstream("cuda.out") << mst_cost << endl;
    ofstream("cuda_timing.out") << elapsed.count() << endl;
    
    // Free GPU memory
    cudaFree(d_edges);
    cudaFree(d_parent);
    cudaFree(d_cheapest);
    cudaFree(d_cost);
    cudaFree(d_edgeUsed);
    cudaFree(d_numComponents);
    
    return 0;
}
