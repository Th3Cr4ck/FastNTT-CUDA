#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define Q 998244353
#define G 3

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

__global__ void fastINTT_simple(uint32_t *d_A, uint32_t N, uint32_t len, uint32_t w_inv, uint32_t q)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N/2) return; // Salir si se lanzan mas de N/2 threads

	uint32_t midlen = len/2;

	uint32_t block = tid / midlen;
	uint32_t k = tid % midlen;
	uint32_t i = block * len;
	
	uint32_t u = d_A[i + k];
	uint32_t v = d_A[i + k + midlen];
	
	// Butterfly inversa
	uint64_t sum = (u + v) % q;
	uint64_t diff = (u + q - v) % q;
	
	uint32_t exponent = (N/len) * k;
	uint64_t twiddle = power(w_inv, exponent, q);
	diff = (diff * twiddle) % q;

	d_A[i + k] = (uint32_t)sum;
	d_A[i + k + midlen] = (uint32_t)diff;
}

void FastNTT_ciclica_iterativa(uint64_t* A, uint64_t* a, uint32_t N, uint32_t w, uint32_t q)
{
    // Copiar entrada
    for (uint32_t i = 0; i < N; i++)
        A[i] = a[i];

    // // --- Bit-reversal permutation ---
    // uint32_t j = 0;
    // for (uint32_t i = 1; i < N; i++) {
    //     uint32_t bit = N >> 1;
    //     while (j & bit) {
    //         j ^= bit;
    //         bit >>= 1;
    //     }
    //     j |= bit;
    //
    //     if (i < j) {
    //         uint64_t temp = A[i];
    //         A[i] = A[j];
    //         A[j] = temp;
    //     }
    // }

    // --- Etapas ---
    for (uint32_t len = 2; len <= N; len <<= 1) {

        uint64_t wlen = power(w, (N / len), q);  // raíz para esta etapa

        for (uint32_t i = 0; i < N; i += len) {

            uint64_t twiddle = 1;

            for (uint32_t k = 0; k < len / 2; k++) {

                uint64_t u = A[i + k];
                uint64_t v = (A[i + k + len/2] * twiddle) % q;

                uint64_t sum = (u + v) % q;
                uint64_t diff = (u + q - v) % q;

                A[i + k] = sum;
                A[i + k + len/2] = diff;

                twiddle = (twiddle * wlen) % q;
            }
        }
    }
}

uint8_t comparar(uint64_t* A, uint64_t* B, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        if (A[i] != B[i]) {
            printf("❌ Error en indice %d: Fast=%ld  Directa=%ld\n",
                   i, A[i], B[i]);
            return 0;
        }
    }
    return 1;
}

void fill_random(uint64_t *a, size_t n)
{
	for (int i=0; i<n; i++)
		a[i] = rand() % Q;
}

int main()
{
    uint32_t sizes[] = {
        1 << 9,   // 512
        1 << 10,  // 1024
        1 << 11   // 2048
    };

    uint8_t num_sizes = sizeof(sizes)/sizeof(sizes[0]);
    uint8_t runs = 5;

    for(int s = 0; s < num_sizes; s++)
    {
        uint32_t N = sizes[s];

        printf("\n=====================================\n");
        printf("Benchmark N = %u\n", N);
        printf("=====================================\n");

        // ---------- Host data ----------
        uint64_t *A = (uint64_t*)malloc(sizeof(uint64_t)*N);
        fill_random(A, N);

        // ---------- Device ----------
        uint64_t *d_A;
        cudaMalloc(&d_A, sizeof(uint64_t)*N);
        cudaMemcpy(d_A, A, sizeof(uint64_t)*N, cudaMemcpyHostToDevice);

        uint64_t w = power(G, (Q-1)/N, Q);

        uint32_t threads = N/2;       
		uint32_t blocks  = 1;

        printf("threads=%u blocks=%u\n", threads, blocks);

        // =====================================================
        // WARM-UP
        // =====================================================
        for(int len=2; len<=N; len*=2)
            fastNTT_simple<<<blocks, threads>>>(d_A, N, len, w, Q);
        cudaDeviceSynchronize();

        // =====================================================
        // BENCHMARK
        // =====================================================
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        for(int r=0; r<runs; r++)
        {
            // restaurar entrada para cada corrida
            cudaMemcpy(d_A, A, sizeof(uint64_t)*N, cudaMemcpyHostToDevice);

            for(int len=2; len<=N; len*=2){
                fastNTT_simple<<<blocks, threads>>>(d_A, N, len, w, Q);
				cudaDeviceSynchronize();
			}
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        // =====================================================
        // MÉTRICAS
        // =====================================================
        double total_time_s = ms / 1000.0;
        double avg_time_s   = total_time_s / runs;
        // double tps          = 1.0 / avg_time_s;

        // throughput estimado
        double bytes = 16.0 * N * log2((double)N);
        double throughput = bytes / (avg_time_s * 1e9);

        printf("Tiempo promedio por NTT: %.6e s\n", avg_time_s);
        printf("Throughput efectivo (estimado): %.3f GB/s\n", throughput);

        // =====================================================
        // VALIDACIÓN
        // =====================================================
        uint64_t *gpu_out = (uint64_t*)malloc(sizeof(uint64_t)*N);
        uint64_t *oro     = (uint64_t*)malloc(sizeof(uint64_t)*N);

        cudaMemcpy(gpu_out, d_A, sizeof(uint64_t)*N, cudaMemcpyDeviceToHost);
        FastNTT_ciclica_iterativa(oro, A, N, w, Q);

        if (!comparar(gpu_out, oro, N))
            printf("❌ Error detectado para N = %u\n", N);
        else
            printf("✅ Verificacion correcta\n");

        // =====================================================
        // CLEANUP
        // =====================================================
        free(oro);
        free(gpu_out);
        free(A);
        cudaFree(d_A);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
