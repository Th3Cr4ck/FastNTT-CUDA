#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define Q 998244353
#define G 3
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

void precomputar_y_copiar_twiddles(uint32_t *stage_offsets, uint32_t N,
                                   uint32_t *twiddles, uint32_t *twiddles_inv,
                                   uint32_t R2, uint32_t n_inv) {

  uint32_t offset = 0;
  uint32_t stage = 0;

  uint32_t w = power(G, (Q - 1) / N, Q);
  uint32_t w_inv = power(w, Q - 2, Q);

  for (int len = 2; len <= N; len <<= 1) {

    stage_offsets[stage] = offset;
    uint32_t mid = len >> 1;

    for (int k = 0; k < mid; k++) {

      uint32_t exponent = (N / len) * k;

      twiddles[offset + k] = mont_mul(power(w, exponent, Q), R2, n_inv);
      twiddles_inv[offset + k] = mont_mul(power(w_inv, exponent, Q), R2, n_inv);
    }

    offset += mid;
    stage++;
  }
}

__global__ void to_montgomery(uint32_t *d_A, uint32_t N, uint32_t R2,
                              uint32_t n_inv) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  // a_mont = a * R mod Q = mont_mul(a, R^2)
  d_A[tid] = mont_mul(d_A[tid], R2, n_inv);
}

__global__ void ntt_block(uint32_t *d_A, uint32_t N, const uint32_t *twiddles,
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

  uint32_t total_stages = __ffs(N);
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

__global__ void ntt_stage(uint32_t *d_A, uint32_t N, const uint32_t *twiddles,
                          const uint32_t *stage_offsets, uint32_t stage,
                          uint32_t len, uint32_t n_inv) {

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N / 2)
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

void FastNTT_ciclica_iterativa(uint32_t *A, uint32_t *a, uint32_t N, uint32_t w,
                               uint32_t q) {
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
  //         uint32_t temp = A[i];
  //         A[i] = A[j];
  //         A[j] = temp;
  //     }
  // }

  // --- Etapas ---
  for (uint32_t len = 2; len <= N; len <<= 1) {

    uint32_t wlen = power(w, (N / len), q); // raíz para esta etapa

    for (uint32_t i = 0; i < N; i += len) {

      uint32_t twiddle = 1;

      for (uint32_t k = 0; k < len / 2; k++) {

        uint32_t u = A[i + k];
        uint32_t v = ((uint64_t)A[i + k + len / 2] * twiddle) % q;

        uint32_t sum = (u + v) % q;
        uint32_t diff = (u + q - v) % q;

        A[i + k] = sum;
        A[i + k + len / 2] = diff;

        twiddle = ((uint64_t)twiddle * wlen) % q;
      }
    }
  }
}

uint8_t comparar(uint32_t *A, uint32_t *B, uint32_t N) {
  for (uint32_t i = 0; i < N; i++) {
    if (A[i] != B[i]) {
      printf("❌ Error en indice %d: Fast=%d  Directa=%d\n", i, A[i], B[i]);
      return 0;
    }
  }
  return 1;
}

void fill_random(uint32_t *a, size_t n) {
  for (int i = 0; i < n; i++)
    a[i] = rand() % Q;
}

int main() {

  uint32_t sizes[] = {
      1 << 9,  // 512
      1 << 10, // 1024
      1 << 11, // 2048
      1 << 12, // 4096
      1 << 13, // 8192
      1 << 14, // 16384
      1 << 15, // 32768
      1 << 16, // 64k
      1 << 17, // 128k
      1 << 18, // 256k
      1 << 19, // 512k
      1 << 20, // 1M
      1 << 21  // 1M
  };

  uint8_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
  uint8_t runs = 255;

  for (int s = 0; s < num_sizes; s++) {

    uint32_t N = sizes[s];

    printf("\n=====================================\n");
    printf("Benchmark N = %u\n", N);
    printf("=====================================\n");

    uint32_t *A = (uint32_t *)malloc(sizeof(uint32_t) * N);
    fill_random(A, N);

    // Precomputar twiddles
    uint32_t n_inv = compute_n_inv(Q);
    uint32_t R2 = (uint64_t)(R % Q) * (R % Q) % Q; // R^2 mod Q
    const uint32_t MAX_TWIDDLES = N - 1;
    uint32_t *twiddles = (uint32_t *)malloc(sizeof(uint32_t) * MAX_TWIDDLES);
    uint32_t *twiddles_inv =
        (uint32_t *)malloc(sizeof(uint32_t) * MAX_TWIDDLES);
    uint32_t stages = 0;
    uint32_t tmp = N;
    while (tmp > 1) {
      stages++;
      tmp >>= 1;
    }
    uint32_t *stage_offsets = (uint32_t *)malloc(sizeof(uint32_t) * stages);
    precomputar_y_copiar_twiddles(stage_offsets, N, twiddles, twiddles_inv, R2,
                                  n_inv);

    // Pasar los datos a memoria global del GPU
    uint32_t *d_A;
    uint32_t *d_twiddles, *d_twiddles_inv;
    uint32_t *d_stage_offsets;

    cudaMalloc(&d_A, sizeof(uint32_t) * N);
    cudaMalloc(&d_twiddles, sizeof(uint32_t) * MAX_TWIDDLES);
    cudaMalloc(&d_twiddles_inv, sizeof(uint32_t) * MAX_TWIDDLES);
    cudaMalloc(&d_stage_offsets, sizeof(uint32_t) * stages);

    cudaMemcpy(d_twiddles, twiddles, sizeof(uint32_t) * MAX_TWIDDLES,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddles_inv, twiddles_inv, sizeof(uint32_t) * MAX_TWIDDLES,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stage_offsets, stage_offsets, sizeof(uint32_t) * stages,
               cudaMemcpyHostToDevice);

    // WARM-UP
    uint32_t threads = 256;
    uint32_t pairs = N / 2;
    uint32_t stage_blocks = (pairs + threads - 1) / threads;
    uint32_t elems_per_block = threads * 2;
    uint32_t blocks = (N + elems_per_block - 1) / elems_per_block;

    uint32_t norm_threads = 256;
    uint32_t norm_blocks = (N + norm_threads - 1) / norm_threads;

    to_montgomery<<<norm_blocks, norm_threads>>>(d_A, N, R2, n_inv);

    ntt_block<<<blocks, threads>>>(d_A, N, d_twiddles, d_stage_offsets, n_inv);

    uint32_t stage = 8;
    for (uint32_t len = 256; len < N; len <<= 1) {

      ntt_stage<<<stage_blocks, threads>>>(d_A, N, d_twiddles, d_stage_offsets,
                                           stage, len, n_inv);
      stage++;
    }

    // BENCHMARK
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int r = 0; r < runs; r++) {

      // restaurar entrada para cada corrida
      cudaMemcpy(d_A, A, sizeof(uint32_t) * N, cudaMemcpyHostToDevice);

      ntt_block<<<blocks, threads>>>(d_A, N, d_twiddles, d_stage_offsets, n_inv);

      uint32_t stage = 8;
      for (uint32_t len = 256; len < N; len <<= 1) {

        ntt_stage<<<stage_blocks, threads>>>(d_A, N, d_twiddles,
                                             d_stage_offsets, stage, len, n_inv);
        stage++;
      }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // MÉTRICAS
    double total_time_s = ms / 1000.0;
    double avg_time_s = total_time_s / runs;

    // throughput estimado
    double bytes = 16.0 * N * stages;
    double throughput = bytes / (avg_time_s * 1e9);

    printf("Tiempo promedio por NTT: %.6e s\n", avg_time_s);
    printf("Throughput efectivo (estimado): %.3f GB/s\n", throughput);

    // VALIDACIÓN
    uint32_t *gpu_out = (uint32_t *)malloc(sizeof(uint32_t) * N);
    uint32_t *oro = (uint32_t *)malloc(sizeof(uint32_t) * N);

    cudaMemcpy(gpu_out, d_A, sizeof(uint32_t) * N, cudaMemcpyDeviceToHost);
    uint32_t w = power(G, (Q - 1) / N, Q);
    FastNTT_ciclica_iterativa(oro, A, N, w, Q);

    if (!comparar(gpu_out, oro, N))
      printf("❌ Error detectado para N = %u\n", N);
    else
      printf("✅ Verificacion correcta\n");

    // CLEANUP
    cudaFree(d_A);
    cudaFree(d_twiddles);
    cudaFree(d_twiddles_inv);
    cudaFree(d_stage_offsets);
    free(twiddles);
    free(twiddles_inv);
    free(oro);
    free(gpu_out);
    free(A);
    free(stage_offsets);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
