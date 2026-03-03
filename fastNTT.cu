#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdint.h>
#include <stdio.h>

#define N_DEF 512
#define Q 998244353
#define G 3
#define W 749
// #define N_DEF	8
// #define Q		3329
// #define W		749
#define MAX_TWIDDLES (N_DEF - 1)

__constant__ uint64_t d_twiddles[MAX_TWIDDLES];
__constant__ uint64_t d_twiddles_inv[MAX_TWIDDLES];

uint64_t power(uint64_t base, uint64_t exp, uint32_t mod) {
  uint64_t res = 1;
  base %= mod;
  while (exp > 0) {
    if (exp % 2 == 1)
      res = (res * base) % mod;
    base = (base * base) % mod;
    exp /= 2;
  }
  return res;
}

__global__ void fastNTT_simple(uint64_t *d_A, uint32_t N, uint32_t stage_offset,
                               uint32_t len, uint32_t q) {
  // *d_A -> Arreglo de entrada
  // N -> Tamaño de d_A
  // len -> Tamaño de un bloque lógico

  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= N / 2)
    return; // Salir si se lanzan mas de N/2 threads

  uint32_t midlen = len / 2; // Número de butterflies por bloque logico

  uint32_t block = tid / midlen; // A qué bloque lógico corresponde el thread
  uint32_t k = tid % midlen;     // Índice del thread dentro del bloque lógico
  uint32_t i = block * len;      // Índice base general del bloque lógico en A

  // Lectura entradas Butterfly
  uint64_t u = d_A[i + k];
  uint64_t v = d_A[i + k + midlen];

  uint64_t twiddle = d_twiddles[stage_offset + k];
  v = (v * twiddle) % q;

  uint64_t sum = (u + v) % q;
  uint64_t diff = (u + q - v) % q;

  d_A[i + k] = sum;
  d_A[i + k + midlen] = diff;
}

__global__ void fastINTT_simple(uint64_t *d_A, uint32_t N,
                                uint32_t stage_offset, uint32_t len,
                                uint32_t q) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= N / 2)
    return; // Salir si se lanzan mas de N/2 threads

  uint32_t midlen = len / 2;

  uint32_t block = tid / midlen;
  uint32_t k = tid % midlen;
  uint32_t i = block * len;

  uint64_t u = d_A[i + k];
  uint64_t v = d_A[i + k + midlen];

  // Butterfly inversa
  uint64_t sum = (u + v) % q;
  uint64_t diff = (u + q - v) % q;

  uint64_t twiddle = d_twiddles_inv[stage_offset + k];
  diff = (diff * twiddle) % q;

  d_A[i + k] = sum;
  d_A[i + k + midlen] = diff;
}

void imprimir_arreglo(uint64_t *A, uint32_t N) {
  for (uint32_t i = 0; i < N; i++) {
    printf("%ld ", A[i]);
  }
  printf("\n");
}

void precomputar_y_copiar_twiddles(uint32_t *stage_offsets) {

  uint64_t twiddles[MAX_TWIDDLES];
  uint64_t twiddles_inv[MAX_TWIDDLES];

  uint32_t offset = 0;
  uint32_t stage = 0;

  uint32_t w = power(G, (Q - 1) / N_DEF, Q);
  uint32_t w_inv = power(w, Q - 2, Q);

  for (int len = 2; len <= N_DEF; len <<= 1) {

    stage_offsets[stage] = offset;

    uint32_t mid = len >> 1;

    for (int k = 0; k < mid; k++) {

      uint32_t exponent = (N_DEF / len) * k;

      twiddles[offset + k] = power(w, exponent, Q);

      twiddles_inv[offset + k] = power(w_inv, exponent, Q);
    }

    offset += mid;
    stage++;
  }

  cudaMemcpyToSymbol(d_twiddles, twiddles, sizeof(uint64_t) * MAX_TWIDDLES);

  cudaMemcpyToSymbol(d_twiddles_inv, twiddles_inv,
                     sizeof(uint64_t) * MAX_TWIDDLES);
}

int main() {
  uint64_t A[N_DEF];

  // Rellenar A con los datos de entrada
  for (uint32_t i = 0; i < N_DEF; i++)
    A[i] = i;

  // Imprimir entrada de ejemplo
  imprimir_arreglo(A, N_DEF);

  // Pasar los datos a memoria global del GPU
  uint64_t *d_A;
  if (cudaMalloc((void **)&d_A, sizeof(uint64_t) * N_DEF) ==
      cudaErrorMemoryAllocation)
    return 1;
  cudaMemcpy(d_A, A, sizeof(uint64_t) * N_DEF, cudaMemcpyHostToDevice);

  // Precomputar twiddles
  uint8_t stages = 0;
  uint32_t tmp = N_DEF;
  while (tmp > 1) {
    tmp >>= 1;
    stages++;
  }

  uint32_t *stage_offsets = (uint32_t *)malloc(sizeof(uint32_t) * stages);
  precomputar_y_copiar_twiddles(stage_offsets);

  // Parámetros para el kernel
  uint32_t threads = N_DEF / 2;

  // Medicion temporal
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // Realizar NTT
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[0], 2, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[1], 4, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[2], 8, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[3], 16, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[4], 32, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[5], 64, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[6], 128, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[7], 256, Q);
  fastNTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[8], 512, Q);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copiar los datos de salida del GPU al Host
  cudaMemcpy(A, d_A, sizeof(uint64_t) * N_DEF, cudaMemcpyDeviceToHost);

  // Imprimir resultado de la NTT
  imprimir_arreglo(A, N_DEF);

  // Realizar INTT
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[8], 512, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[7], 256, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[6], 128, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[5], 64, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[4], 32, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[3], 16, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[2], 8, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[1], 4, Q);
  fastINTT_simple<<<1, threads>>>(d_A, N_DEF, stage_offsets[0], 2, Q);

  // Copiar los datos de salida del GPU al Host
  cudaMemcpy(A, d_A, sizeof(uint64_t) * N_DEF, cudaMemcpyDeviceToHost);

  // Escalar resultado por N^-1
  uint32_t N_inv = power(N_DEF, Q - 2, Q);
  for (int i = 0; i < N_DEF; i++)
    A[i] = (A[i] * N_inv) % Q;

  // Imprimir resultado de la INTT
  imprimir_arreglo(A, N_DEF);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  printf("La NTT tardó: %f ms\n", ms);

  cudaFree(d_A);
  free(stage_offsets);
}
