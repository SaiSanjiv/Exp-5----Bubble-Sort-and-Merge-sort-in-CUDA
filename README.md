# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```
%%writefile sorting.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <algorithm> // Still useful for host-side std::min if needed, but will use device min for kernels

// Kernel for Bubble Sort
__global__ void bubbleSortKernel(int *d_arr, int n) {
    int temp;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread handles a single bubble sort pass
    for(int i=0;i<n-1;i++){
      if(idx < n-1-i){
        if(d_arr[idx] > d_arr[idx+1]){
          //Swap
          temp = d_arr[idx];
          d_arr[idx] = d_arr[idx+1];
          d_arr[idx+1] = temp;
        }
      }
        __syncthreads(); // Synchronize threads after each pass
    }
}

// Device function for merging arrays (now uses d_temp as auxiliary buffer)
__device__ void merge(int *d_arr, int *d_temp, int left, int mid, int right) {
   int i,j,k;
   int n1=mid-left+1;
   int n2 = right-mid;

   // Copy left half to d_temp
   for(i=0;i<n1;i++)
     d_temp[i] = d_arr[left+i];
   // Copy right half to d_temp (offset by n1)
   for(j=0;j<n2;j++)
     d_temp[n1+j] = d_arr[mid+1+j];

   i=0;         // Initial index of first sub-array in d_temp
   j=n1;        // Initial index of second sub-array in d_temp
   k=left;      // Initial index of merged sub-array in d_arr

   while(i<n1 && j<n1+n2){
    if(d_temp[i] <= d_temp[j]){
      d_arr[k] = d_temp[i];
      i++;
    }
    else{
      d_arr[k] = d_temp[j];
      j++;
    }
    k++;
   }

   while(i<n1){
    d_arr[k] = d_temp[i];
    i++;
    k++;
   }

  while(j<n1+n2){
    d_arr[k] = d_temp[j];
    j++;
    k++;
  }
}

// Kernel for Merge Sort
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n) {
    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1); // Changed std::min to device min

            merge(d_arr, d_temp, left, mid, right); // Pass d_temp to device merge
            left += 2 * size;
        }
        // This copy back is not needed if merge function directly merges into d_arr
        // The typical merge sort on GPU uses ping-pong buffers, alternating source/destination
        // For this single-threaded kernel, the merge function modifies d_arr in place after using d_temp
        // as auxiliary, so no full array copy is needed here after each pass.
        // However, if the merge operation were to merge from d_arr to d_temp, then d_temp would need to be copied back.
        // Given the current merge function logic, the d_arr is modified directly.
        // The original `for` loops for `d_temp` copy were also slightly misplaced for standard merge sort logic.
        // Let's simplify and assume `merge` does the final write to `d_arr`.
    }
}

// Host function for merging arrays
void mergeHost(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// Bubble Sort on GPU
void bubbleSort(int *arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bubbleSortKernel<<<1, n>>>(d_arr, n);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds); // Fixed format specifier
}

// Merge Sort on GPU
void mergeSort(int *arr, int n) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mergeSortKernel<<<1, 1>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

// Bubble Sort on CPU
void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration duration = end - start;
    printf("Bubble Sort (CPU) took %ld milliseconds\n", duration.count());
}

// Merge Sort on CPU
void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size - 1;
            int right = std::min(left + 2 * size - 1, n - 1); // std::min is fine here as it's a host function

            mergeHost(arr, left, mid, right); // Call host merge
            left += 2 * size;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> duration = end - start;
    printf("Merge Sort (CPU) took %lf milliseconds\n", duration.count());
}

// Print array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

// Main function
int main() {
    int n = 10000; // Increase for larger datasets
    int *arr = (int*)malloc(n * sizeof(int));

    // Generating random array
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("Original array: \n");
    printArray(arr, n);

    // Bubble Sort CPU
    bubbleSortCPU(arr, n);
    printf("Sorted array using Bubble Sort (CPU): \n");
    printArray(arr, n);

    // Generating random array again for GPU
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Bubble Sort GPU
    bubbleSort(arr, n);
    printf("Sorted array using Bubble Sort (GPU): \n");
    printArray(arr, n);

    // Generating random array again for Merge Sort
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    printf("Original array: \n");
    printArray(arr, n);

    // Merge Sort CPU
    mergeSortCPU(arr, n);
    printf("Sorted array using Merge Sort (CPU): \n");
    printArray(arr, n);

    // Generating random array again for GPU
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;
    }

    // Merge Sort GPU
    mergeSort(arr, n);
    printf("Sorted array using Merge Sort (GPU): \n");
    printArray(arr, n);

    free(arr);
    return 0;
}

```

## OUTPUT:
<img width="1757" height="328" alt="image" src="https://github.com/user-attachments/assets/9495e3d8-3c68-457a-ba70-aa7be744931b" />


## RESULT:
Thus, the program has been executed using CUDA to implement Bubble Sort and Merge Sort.
