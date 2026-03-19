#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define Q 998244353
#define G 3

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

void precomputar_y_copiar_twiddles(uint32_t n, uint32_t *stage_offsets,
                                   uint32_t *twiddles, uint32_t *twiddles_inv) {

  uint32_t offset = 0;
  uint32_t stage = 0;

  uint32_t w = power(G, (Q - 1) / n, Q);
  uint32_t w_inv = power(w, Q - 2, Q);

  for (int len = 2; len <= n; len <<= 1) {

    stage_offsets[stage] = offset;
    uint32_t mid = len >> 1;

    for (int k = 0; k < mid; k++) {

      uint32_t exponent = (n / len) * k;

      twiddles[offset + k] = power(w, exponent, Q);

      twiddles_inv[offset + k] = power(w_inv, exponent, Q);
    }

    offset += mid;
    stage++;
  }
}

__global__ void ntt_stage(uint32_t *d_A, uint32_t n, const uint32_t *twiddles,
                          const uint32_t *stage_offsets, uint32_t stage,
                          uint32_t len) {

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n / 2)
    return;

  uint32_t j = tid & (len - 1);
  uint32_t group = tid >> stage;
  uint32_t pos = (group << (stage + 1)) + j;

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

__global__ void ntt_warp(uint32_t *d_A, uint32_t n, const uint32_t *twiddles,
                         const uint32_t *stage_offsets) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n / 2)
    return;

  uint32_t stage = 0;

  for (uint32_t len = 1; len < 32 && len < n; len <<= 1) {

    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

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

__global__ void intt_stage(uint32_t *d_A, uint32_t n,
                           const uint32_t *twiddles_inv,
                           const uint32_t *stage_offsets, uint32_t stage,
                           uint32_t len) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t pairs = n >> 1;

  if (tid >= pairs)
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
  uint32_t t = ((uint64_t)diff * w) % Q;

  d_A[pos] = sum;
  d_A[pos + len] = t;
}

__global__ void intt_normalize(uint32_t *d_A, uint32_t n, uint32_t n_inv) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n)
    return;

  d_A[tid] = ((uint64_t)d_A[tid] * n_inv) % Q;
}

__global__ void intt_warp(uint32_t *d_A, uint32_t n,
                          const uint32_t *twiddles_inv,
                          const uint32_t *stage_offsets, uint32_t stage_start) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n / 2)
    return;

  uint32_t stage = stage_start;

  for (uint32_t len = 32; len > 0; len >>= 1, stage--) {

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
    uint32_t t = ((uint64_t)diff * w) % Q;

    d_A[pos] = sum;
    d_A[pos + len] = t;
  }
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
    const uint32_t MAX_TWIDDLES = N - 1;
    uint32_t twiddles[MAX_TWIDDLES];
    uint32_t twiddles_inv[MAX_TWIDDLES];
    uint32_t stages = 0;
    uint32_t tmp = N;
    while (tmp > 1) {
      stages++;
      tmp >>= 1;
    }
    uint32_t *stage_offsets = (uint32_t *)malloc(sizeof(uint32_t) * stages);
    precomputar_y_copiar_twiddles(N, stage_offsets, twiddles, twiddles_inv);

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
    uint32_t blocks = (pairs + threads - 1) / threads;
    uint32_t warp_threads = 256;
    uint32_t warp_blocks = ((N / 2) + warp_threads - 1) / warp_threads;
    ntt_warp<<<warp_blocks, warp_threads>>>(d_A, N, d_twiddles,
                                            d_stage_offsets);

    uint32_t stage = 5;
    for (uint32_t len = 32; len < N; len <<= 1) {

      ntt_stage<<<blocks, threads>>>(d_A, N, d_twiddles, d_stage_offsets, stage,
                                     len);
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

      ntt_warp<<<warp_blocks, warp_threads>>>(d_A, N, d_twiddles,
                                              d_stage_offsets);

      uint32_t stage = 5;
      for (uint32_t len = 32; len < N; len <<= 1) {

        ntt_stage<<<blocks, threads>>>(d_A, N, d_twiddles, d_stage_offsets,
                                       stage, len);
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
    free(oro);
    free(gpu_out);
    free(A);
    free(stage_offsets);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
