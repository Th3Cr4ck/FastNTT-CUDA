#include <cstdint>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdint.h>
#include <stdio.h>
#include <texture_indirect_functions.h>

// #define LEN_TINY
#define LEN_LARGE

#ifdef LEN_TINY
#define N_DEF 64
#define Q 3329
#elif defined(LEN_LARGE)
#define N_DEF 2048
#define Q 998244353
#else
#error "Debes definir LEN_8 o LEN_LARGE"
#endif

#define G 3
#define MAX_TWIDDLES (N_DEF - 1)
#define R_BITS 32
#define R (1ULL << R_BITS)

uint32_t compute_n_inv(uint32_t n) {
  // n* x ≡ 1 (mod 2^32), resuelto iterativamente
  uint32_t x = 1;
  for (int i = 0; i < 31; i++)
    x *= 2 - n * x;
  return -x; // Retorna -Q^(-1) mod 2^32
}

__host__ __device__ __forceinline__ uint32_t mont_reduce(uint64_t x,
                                                         uint32_t n_inv) {
  uint32_t m = (uint32_t)x * n_inv;         // m = x * (-Q^{-1}) mod 2^32
  uint32_t t = (x + (uint64_t)m * Q) >> 32; // t = (x + m*Q) / 2^32
  return t >= Q ? t - Q : t;
}

__host__ __device__ __forceinline__ uint32_t mont_mul(uint32_t a, uint32_t b,
                                                      uint32_t n_inv) {
  return mont_reduce((uint64_t)a * b, n_inv);
}

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
                                   uint32_t *twiddles_inv, uint32_t R2, uint32_t n_inv) {

  uint32_t offset = 0;
  uint32_t stage = 0;

  uint32_t w = power(G, (Q - 1) / N_DEF, Q);
  uint32_t w_inv = power(w, Q - 2, Q);

  for (int len = 2; len <= N_DEF; len <<= 1) {

    stage_offsets[stage] = offset;
    uint32_t mid = len >> 1;

    for (int k = 0; k < mid; k++) {

      uint32_t exponent = (N_DEF / len) * k;

      twiddles[offset + k] = mont_mul(power(w, exponent, Q), R2, n_inv);
      twiddles_inv[offset + k] = mont_mul(power(w_inv, exponent, Q), R2, n_inv);
    }

    offset += mid;
    stage++;
  }
}

__global__ void to_montgomery(uint32_t *d_A, uint32_t R2, uint32_t n_inv) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N_DEF)
    return;
  // a_mont = a * R mod Q = mont_mul(a, R^2)
  d_A[tid] = mont_mul(d_A[tid], R2, n_inv);
}

__global__ void ntt_block(uint32_t *d_A, const uint32_t *twiddles,
                          const uint32_t *stage_offsets, uint32_t n_inv) {

  __shared__ uint32_t s_A[512];

  uint32_t tid = threadIdx.x; // 0 a 255
  uint32_t base =
      blockIdx.x *
      (blockDim.x << 1); // Indice de inicio del bloque logico de 512 elementos

  // Cargar a shared
  s_A[tid] = d_A[base + tid];
  s_A[tid + blockDim.x] = d_A[base + tid + blockDim.x];

  __syncthreads();

  uint32_t total_stages = __ffs(N_DEF);
  uint32_t stage = 0;

  // Warp stages
  for (; stage < 5 && stage < total_stages; stage++) {

    uint32_t len = 1 << stage;

    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos];
    uint32_t v = s_A[pos + len];

    uint32_t w = twiddles[stage_offsets[stage] + j];
    uint32_t t = mont_mul(v, w, n_inv);

    uint32_t sum = u + t;
    if (sum >= Q)
      sum -= Q;

    uint32_t diff = u + Q - t;
    if (diff >= Q)
      diff -= Q;

    s_A[pos] = sum;
    s_A[pos + len] = diff;
  }

  __syncthreads();

  // Intra block
  for (; stage < 8 && stage < total_stages; stage++) {

    uint32_t len = 1 << stage;

    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos];
    uint32_t v = s_A[pos + len];

    uint32_t w = twiddles[stage_offsets[stage] + j];
    uint32_t t = mont_mul(v, w, n_inv);

    uint32_t sum = u + t;
    if (sum >= Q)
      sum -= Q;

    uint32_t diff = u + Q - t;
    if (diff >= Q)
      diff -= Q;

    s_A[pos] = sum;
    s_A[pos + len] = diff;

    __syncthreads();
  }

  // Escribir de regreso
  d_A[base + tid] = s_A[tid];
  d_A[base + tid + blockDim.x] = s_A[tid + blockDim.x];
}

__global__ void ntt_stage(uint32_t *d_A, const uint32_t *twiddles,
                          const uint32_t *stage_offsets, uint32_t stage,
                          uint32_t len, uint32_t n_inv) {

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N_DEF / 2)
    return;

  uint32_t j = tid & (len - 1);
  uint32_t group = tid >> stage;
  uint32_t pos = (group << (stage + 1)) + j;

  uint32_t u = d_A[pos];
  uint32_t v = d_A[pos + len];

  uint32_t w = twiddles[stage_offsets[stage] + j];
  uint32_t t = mont_mul(v, w, n_inv);

  uint32_t sum = u + t;
  if (sum >= Q)
    sum -= Q;

  uint32_t diff = u + Q - t;
  if (diff >= Q)
    diff -= Q;

  d_A[pos] = sum;
  d_A[pos + len] = diff;
}

__global__ void intt_stage(uint32_t *d_A, const uint32_t *twiddles_inv,
                           const uint32_t *stage_offsets, uint32_t stage,
                           uint32_t len, uint32_t n_inv) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N_DEF >> 1)
    return;

  uint32_t j = tid & (len - 1);
  uint32_t group = tid >> stage;
  uint32_t pos = (group << (stage + 1)) + j;

  uint32_t u = d_A[pos];
  uint32_t v = d_A[pos + len];

  uint32_t sum = u + v;
  if (sum >= Q)
    sum -= Q;

  uint32_t diff = u + Q - v;
  if (diff >= Q)
    diff -= Q;

  uint32_t w = twiddles_inv[stage_offsets[stage] + j];
  uint32_t t = mont_mul(diff, w, n_inv);

  d_A[pos] = sum;
  d_A[pos + len] = t;
}

__global__ void intt_block(uint32_t *d_A, const uint32_t *twiddles_inv,
                           const uint32_t *stage_offsets, uint32_t n_inv) {

  __shared__ uint32_t s_A[512];

  uint32_t tid = threadIdx.x;
  uint32_t base = blockIdx.x * (blockDim.x << 1);

  // Cargar a shared
  s_A[tid] = d_A[base + tid];
  s_A[tid + blockDim.x] = d_A[base + tid + blockDim.x];

  __syncthreads();

  // Intra block (stages grandes)
  for (uint32_t loop_stage = 0; loop_stage < 3; loop_stage++) {

    uint32_t stage = 7 - loop_stage;
    uint32_t len = 1 << stage;

    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos];
    uint32_t v = s_A[pos + len];

    uint32_t sum = u + v;
    if (sum >= Q)
      sum -= Q;

    uint32_t diff = u + Q - v;
    if (diff >= Q)
      diff -= Q;

    uint32_t w = twiddles_inv[stage_offsets[stage] + j];
    uint32_t t = mont_mul(diff, w, n_inv);

    s_A[pos] = sum;
    s_A[pos + len] = t;

    __syncthreads();
  }

  // Warp stages (stages pequeños)
  for (uint32_t loop_stage = 3; loop_stage < 8; loop_stage++) {

    uint32_t stage = 7 - loop_stage;
    uint32_t len = 1 << stage;

    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos];
    uint32_t v = s_A[pos + len];

    uint32_t sum = u + v;
    if (sum >= Q)
      sum -= Q;

    uint32_t diff = u + Q - v;
    if (diff >= Q)
      diff -= Q;

    uint32_t w = twiddles_inv[stage_offsets[stage] + j];
    uint32_t t = mont_mul(diff, w, n_inv);

    s_A[pos] = sum;
    s_A[pos + len] = t;
  }

  __syncthreads();

  // Escribir de regreso
  d_A[base + tid] = s_A[tid];
  d_A[base + tid + blockDim.x] = s_A[tid + blockDim.x];
}

__global__ void intt_normalize_from_mont(uint32_t *d_A, uint32_t mont_n_inv,
                                          uint32_t N_inv_mont) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_DEF) return;

    // Sale de Montgomery, luego multiplica por N^{-1} (también en Mont.)
    uint32_t val = mont_reduce((uint64_t)d_A[tid], mont_n_inv);
    d_A[tid] = mont_mul(val, N_inv_mont, mont_n_inv);
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
  uint32_t n_inv = compute_n_inv(Q);
  uint32_t R2 = (uint64_t)(R % Q) * (R % Q) % Q; // R^2 mod Q
  uint32_t twiddles[MAX_TWIDDLES];
  uint32_t twiddles_inv[MAX_TWIDDLES];
  uint32_t total_stages = 0;
  uint32_t tmp = N_DEF;
  while (tmp > 1) {
    total_stages++;
    tmp >>= 1;
  }
  uint32_t *stage_offsets = (uint32_t *)malloc(sizeof(uint32_t) * total_stages);
  precomputar_y_copiar_twiddles(stage_offsets, twiddles, twiddles_inv, R2, n_inv);

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
  uint32_t stage_blocks =
      (pairs + threads - 1) / threads; // blocks = ceil(butterflies/threads)
  uint32_t elems_per_block = threads * 2;
  uint32_t blocks =
      (N_DEF + elems_per_block - 1) / elems_per_block; // blocks = ceil(N/512)

  uint32_t norm_threads = 256;
  uint32_t norm_blocks = (N_DEF + norm_threads - 1) / norm_threads;

  // Medicion temporal
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  /* ------------------ NTT ----------------- */

  to_montgomery<<<norm_blocks, norm_threads>>>(d_A, R2, n_inv);

  ntt_block<<<blocks, threads>>>(d_A, d_twiddles, d_stage_offsets, n_inv);

  uint32_t stage = 8;
  for (uint32_t len = 256; len < N_DEF; len <<= 1) {

    ntt_stage<<<stage_blocks, threads>>>(d_A, d_twiddles, d_stage_offsets,
                                         stage, len, n_inv);
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
  for (uint32_t len = N_DEF >> 1; len >= 256; len >>= 1, stage--) {
    intt_stage<<<stage_blocks, threads>>>(d_A, d_twiddles_inv, d_stage_offsets,
                                          stage, len, n_inv);
  }

  intt_block<<<stage_blocks, threads>>>(d_A, d_twiddles_inv, d_stage_offsets, n_inv);

  // Normalizar intt
  uint32_t N_inv = power(N_DEF, Q - 2, Q);
  uint32_t N_inv_mont = mont_mul(N_inv, R2, n_inv);
  intt_normalize_from_mont<<<norm_blocks, norm_threads>>>(d_A, n_inv, N_inv_mont);

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
