#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdint.h>
#include <stdio.h>

#define N_DEF	512
#define Q		998244353
#define G		3
#define W		749
// #define N_DEF	8
// #define Q		3329
// #define W		749

__host__ __device__ 
uint64_t power(uint64_t base, uint64_t exp, uint32_t mod) {
	uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

__global__
void fastNTT_simple(uint64_t *d_A, uint32_t N, uint32_t len, uint32_t w, uint32_t q)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N/2) return; // Salir si se lanzan mas de N/2 threads

	uint32_t midlen = len/2;

	uint32_t block = tid / midlen;
	uint32_t k = tid % midlen;
	uint32_t i = block * len;
	
	// Lectura entradas Butterfly
	uint64_t u = d_A[i + k];
	uint64_t v = d_A[i + k + midlen];
	
	uint32_t exponent = (N/len) * k;
	uint64_t twiddle = power(w, exponent, q);
	v = (v * twiddle) % q;

	uint64_t sum = (u + v) % q;
	uint64_t diff = (u + q - v) % q;
	
	d_A[i + k] = sum;
	d_A[i + k + midlen] = diff;
}

__global__ void fastINTT_simple(uint64_t *d_A, uint32_t N, uint32_t len, uint32_t w_inv, uint32_t q)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N/2) return; // Salir si se lanzan mas de N/2 threads

	uint32_t midlen = len/2;

	uint32_t block = tid / midlen;
	uint32_t k = tid % midlen;
	uint32_t i = block * len;
	
	uint64_t u = d_A[i + k];
	uint64_t v = d_A[i + k + midlen];
	
	// Butterfly inversa
	uint64_t sum = (u + v) % q;
	uint64_t diff = (u + q - v) % q;
	
	uint32_t exponent = (N/len) * k;
	uint64_t twiddle = power(w_inv, exponent, q);
	diff = (diff * twiddle) % q;

	d_A[i + k] = sum;
	d_A[i + k + midlen] = diff;
}

void imprimir_arreglo(uint64_t *A, uint32_t N)
{
	for (uint32_t i=0; i<N; i++) {
		printf("%ld ", A[i]);
	}
	printf("\n");
}

int main()
{
	uint64_t A[N_DEF];
	
	// Rellenar A con los datos de entrada
	for (uint32_t i=0; i<N_DEF; i++)
		A[i] = i;

	// Imprimir entrada de ejemplo
	imprimir_arreglo(A, N_DEF);

	// Pasar los datos a memoria global que pueda usar el GPU
	uint64_t *d_A;
	if (cudaMalloc((void**)&d_A, sizeof(uint64_t)*N_DEF) == cudaErrorMemoryAllocation) return 1;
	cudaMemcpy(d_A, A, sizeof(uint64_t)*N_DEF, cudaMemcpyHostToDevice);

	uint32_t threads = N_DEF / 2;
	uint32_t blockSize = N_DEF;
	uint32_t gridSize = 1;
	
    uint32_t w = power(G, (Q-1)/N_DEF, Q);

	// Medicion temporal
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);

	// Realizar NTT
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 2, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 4, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 8, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 16, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 32, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 64, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 128, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 256, w, Q);
	fastNTT_simple<<<1,threads>>>(d_A, blockSize, 512, w, Q);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Copiar los datos de salida de la memoria global del GPU
	cudaMemcpy(A, d_A, sizeof(uint64_t)*N_DEF, cudaMemcpyDeviceToHost);

	// Imprimir resultado de la NTT
	imprimir_arreglo(A, N_DEF);

	// Realizar INTT
	uint32_t w_inv = power(w, Q-2, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 512, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 256, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 128, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 64, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 32, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 16, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 8, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 4, w_inv, Q);
	fastINTT_simple<<<1,threads>>>(d_A,  blockSize, 2, w_inv, Q);
	
	// Copiar los datos de salida de la memoria global del GPU
	cudaMemcpy(A, d_A, sizeof(uint64_t)*N_DEF, cudaMemcpyDeviceToHost);

	// Escalar resultado por N^-1
	uint32_t N_inv = power(N_DEF, Q-2, Q);
	for (int i=0; i<N_DEF; i++)
		A[i] = (A[i] * N_inv) % Q;

	// Imprimir resultado de la INTT
	imprimir_arreglo(A, N_DEF);

	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	
	printf("La NTT tardó: %f ms\n", ms);

	cudaFree(d_A);
}


