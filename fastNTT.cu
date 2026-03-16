#include <cstdint>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdint.h>
#include <stdio.h>
#include <texture_indirect_functions.h>

// #define LEN_64
#define LEN_512

#ifdef LEN_64
#define N_DEF 64
#define Q 3329
#elif defined(LEN_512)
#define N_DEF 8192
#define Q 998244353
#else
#error "Debes definir LEN_8 o LEN_512"
#endif

#define G 3
#define MAX_TWIDDLES (N_DEF - 1)

uint32_t power(uint32_t base, uint32_t exp, uint32_t mod) {
  uint64_t res = 1;
  base %= mod;
  while (exp > 0) {
    if (exp % 2 == 1)
      res = (res * base) % mod;
    base = ((uint64_t)base * base) % mod;
    exp /= 2;
  }
  return (uint32_t)res;
}

void imprimir_arreglo(uint32_t *A, uint32_t N) {
  for (uint32_t i = 0; i < N; i++) {
    printf("%d ", A[i]);
  }
  printf("\n");
}

void precomputar_y_copiar_twiddles(uint32_t *stage_offsets, uint32_t *twiddles,
                                   uint32_t *twiddles_inv) {

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
}

__global__ void ntt_stage(uint32_t *d_A, const uint32_t *twiddles,
                          const uint32_t *stage_offsets, uint32_t stage,
                          uint32_t len) {

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t pairs = N_DEF >> 1;

  if (tid >= pairs)
    return;

  uint32_t j = tid % len;
  uint32_t group = tid / len;
  uint32_t pos = group * (2 * len) + j;

  uint32_t u = d_A[pos];
  uint32_t v = d_A[pos + len];

  uint32_t w = twiddles[stage_offsets[stage] + j];
  uint32_t t = ((uint64_t)v * w) % Q;

  uint32_t sum = u + t;
  if (sum >= Q)
    sum -= Q;

  uint32_t diff = u + Q - t;
  if (diff >= Q)
    diff -= Q;

  d_A[pos] = sum;
  d_A[pos + len] = diff;
}

__global__ void ntt_warp(uint32_t *d_A, const uint32_t *twiddles,
                         const uint32_t *stage_offsets) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t n = N_DEF;

  if (tid >= n / 2)
    return;

  uint32_t stage = 0;

  for (uint32_t len = 1; len < 32 && len < n; len <<= 1) {

    uint32_t group = tid / len;
    uint32_t j = tid % len;

    uint32_t pos = group * (2 * len) + j;

    uint32_t u = d_A[pos];
    uint32_t v = d_A[pos + len];

    uint32_t w = twiddles[stage_offsets[stage] + j];
    uint32_t t = ((uint64_t)v * w) % Q;

    uint32_t sum = u + t;
    if (sum >= Q)
      sum -= Q;

    uint32_t diff = u + Q - t;
    if (diff >= Q)
      diff -= Q;

    d_A[pos] = sum;
    d_A[pos + len] = diff;

    stage++;
  }
}

__global__ void intt_stage(uint32_t *d_A, const uint32_t *twiddles_inv,
                           const uint32_t *stage_offsets, uint32_t stage,
                           uint32_t len) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t pairs = N_DEF >> 1;

  if (tid >= pairs)
    return;

  uint32_t j = tid % len;
  uint32_t group = tid / len;

  uint32_t pos = group * (2 * len) + j;

  uint32_t u = d_A[pos];
  uint32_t v = d_A[pos + len];

  uint32_t sum = u + v;
  if (sum >= Q)
    sum -= Q;

  uint32_t diff = u + Q - v;
  if (diff >= Q)
    diff -= Q;

  uint32_t w = twiddles_inv[stage_offsets[stage] + j];
  uint32_t t = ((uint64_t)diff * w) % Q;

  d_A[pos] = sum;
  d_A[pos + len] = t;
}

__global__ void intt_normalize(uint32_t *d_A, uint32_t n_inv) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N_DEF)
    return;

  d_A[tid] = ((uint64_t)d_A[tid] * n_inv) % Q;
}

__global__ void intt_warp(uint32_t *d_A, const uint32_t *twiddles_inv,
                          const uint32_t *stage_offsets, uint32_t stage_start) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t n = N_DEF;

  if (tid >= n / 2)
    return;

  uint32_t stage = stage_start;

  for (uint32_t len = 32; len > 0; len >>= 1, stage--) {

    uint32_t group = tid / len;
    uint32_t j = tid % len;

    uint32_t pos = group * (2 * len) + j;

    uint32_t u = d_A[pos];
    uint32_t v = d_A[pos + len];

    uint32_t sum = u + v;
    if (sum >= Q)
      sum -= Q;

    uint32_t diff = u + Q - v;
    if (diff >= Q)
      diff -= Q;

    uint32_t w = twiddles_inv[stage_offsets[stage] + j];
    uint32_t t = ((uint64_t)diff * w) % Q;

    d_A[pos] = sum;
    d_A[pos + len] = t;
  }
}

int main() {

  /* ------------------ Init ----------------- */

  uint32_t A[N_DEF];

  // Rellenar A con los datos de entrada
  for (uint32_t i = 0; i < N_DEF; i++)
    A[i] = i;

  // Imprimir entrada de ejemplo
  imprimir_arreglo(A, N_DEF);

  // Precomputar twiddles
  uint32_t twiddles[MAX_TWIDDLES];
  uint32_t twiddles_inv[MAX_TWIDDLES];
  uint32_t total_stages = 0;
  uint32_t tmp = N_DEF;
  while (tmp > 1) {
    total_stages++;
    tmp >>= 1;
  }
  uint32_t *stage_offsets = (uint32_t *)malloc(sizeof(uint32_t) * total_stages);
  precomputar_y_copiar_twiddles(stage_offsets, twiddles, twiddles_inv);

  // Pasar los datos a memoria global del GPU
  uint32_t *d_A;
  uint32_t *d_twiddles, *d_twiddles_inv;
  uint32_t *d_stage_offsets;

  cudaMalloc(&d_A, sizeof(uint32_t) * N_DEF);
  cudaMalloc(&d_twiddles, sizeof(uint32_t) * MAX_TWIDDLES);
  cudaMalloc(&d_twiddles_inv, sizeof(uint32_t) * MAX_TWIDDLES);
  cudaMalloc(&d_stage_offsets, sizeof(uint32_t) * total_stages);

  cudaMemcpy(d_A, A, sizeof(uint32_t) * N_DEF, cudaMemcpyHostToDevice);
  cudaMemcpy(d_twiddles, twiddles, sizeof(uint32_t) * MAX_TWIDDLES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_twiddles_inv, twiddles_inv, sizeof(uint32_t) * MAX_TWIDDLES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_stage_offsets, stage_offsets, sizeof(uint32_t) * total_stages,
             cudaMemcpyHostToDevice);

  uint32_t threads = 256;
  uint32_t pairs = N_DEF / 2;
  uint32_t blocks = (pairs + threads - 1) / threads;

  // Medicion temporal
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  /* ------------------ NTT ----------------- */

  uint32_t warp_threads = 256;
  uint32_t warp_blocks = ((N_DEF / 2) + warp_threads - 1) / warp_threads;
  ntt_warp<<<warp_blocks, warp_threads>>>(d_A, d_twiddles, d_stage_offsets);

  uint32_t stage = 5;
  for (uint32_t len = 32; len < N_DEF; len <<= 1) {

    ntt_stage<<<blocks, threads>>>(d_A, d_twiddles, d_stage_offsets, stage,
                                   len);
    stage++;
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copiar los datos de salida del GPU al Host
  cudaMemcpy(A, d_A, sizeof(uint32_t) * N_DEF, cudaMemcpyDeviceToHost);

  // Imprimir resultado de la NTT
  imprimir_arreglo(A, N_DEF);

  /* ------------------ INTT ----------------- */

  stage = total_stages - 1;
  for (uint32_t len = N_DEF >> 1; len > 32; len >>= 1) {

    intt_stage<<<blocks, threads>>>(d_A, d_twiddles_inv, d_stage_offsets, stage,
                                    len);
    stage--;
  }

  intt_warp<<<warp_blocks, warp_threads>>>(d_A, d_twiddles_inv, d_stage_offsets,
                                           stage);

  // Normalizar intt
  uint32_t N_inv = power(N_DEF, Q - 2, Q);
  uint32_t norm_threads = 256;
  uint32_t norm_blocks = (N_DEF + norm_threads - 1) / norm_threads;
  intt_normalize<<<norm_blocks, norm_threads>>>(d_A, N_inv);

  // Copiar los datos de salida del GPU al Host
  cudaMemcpy(A, d_A, sizeof(uint32_t) * N_DEF, cudaMemcpyDeviceToHost);

  // Imprimir resultado de la INTT
  imprimir_arreglo(A, N_DEF);

  /* ------------------ Mediciones ----------------- */

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("La NTT tardó: %f ms\n", ms);

  free(stage_offsets);
  cudaFree(d_A);
  cudaFree(d_twiddles);
  cudaFree(d_twiddles_inv);
  cudaFree(d_stage_offsets);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
