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

void imprimir_arreglo(uint64_t *A, uint32_t N) {
  for (uint32_t i = 0; i < N; i++) {
    printf("%ld ", A[i]);
  }
  printf("\n");
}

void precomputar_y_copiar_twiddles(uint32_t *stage_offsets, uint64_t *twiddles,
                                   uint64_t *twiddles_inv) {

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

__global__ void ntt_kernel(uint64_t *d_A, const uint64_t *twiddles,
                           const uint32_t *stage_offsets) {

  uint32_t tid = threadIdx.x;
  uint32_t n = N_DEF;

  uint32_t stage = 0;

  for (uint32_t len = 1; len < n; len <<= 1) {

    uint32_t butterflies = n >> 1;


    for (uint32_t b = tid; b < butterflies; b += blockDim.x) {

      // cada butterfly tiene:
      uint32_t group =
          b / len; // A qué bloque (grupo) lógico corresponde el thread
      uint32_t j = b % len; // Índice del thread dentro del bloque lógico

      uint32_t base =
          group * (2 * len); // Índice base del bloque (grupo) lógico
      uint32_t pos = base + j;

      uint64_t u = d_A[pos];
      uint64_t v = d_A[pos + len];

      uint64_t w = twiddles[stage_offsets[stage] + j];

      uint64_t t = (v * w) % Q;

      uint64_t sum = u + t;
      if (sum >= Q)
        sum -= Q;

      uint64_t diff = u + Q - t;
      if (diff >= Q)
        diff -= Q;

      d_A[pos] = sum;
      d_A[pos + len] = diff;
    }

    stage++;
    __syncthreads();
  }

}

__global__ void intt_kernel(uint64_t *d_A, const uint64_t *twiddles_inv,
                            const uint32_t *stage_offsets,
                            const uint32_t n_inv) {

  const uint32_t tid = threadIdx.x;
  const uint32_t n = N_DEF;
  const uint32_t total_stages = __ffs(n) - 1; //log2(n)

  uint32_t loop_stage = 0;

  for (uint32_t len = n >> 1; len >= 1; len >>= 1, loop_stage++) {

    uint32_t stage = total_stages - 1 - loop_stage;
    uint32_t butterflies = n >> 1;

    for (uint32_t b = tid; b < butterflies; b += blockDim.x) {

      uint32_t group = b / len; // A qué bloque (grupo) lógico corresponde el thread
      uint32_t j = b % len; // Índice del thread dentro del bloque lógico

      uint32_t base = group * (2 * len); // Índice base del bloque (grupo) lógico
      uint32_t pos = base + j;

      uint64_t u = d_A[pos];
      uint64_t v = d_A[pos + len];

      uint64_t sum = u + v;
      if (sum >= Q)
        sum -= Q;

      uint64_t diff = u + Q - v;
      if (diff >= Q)
        diff -= Q;

      uint64_t w = twiddles_inv[stage_offsets[stage] + j];
      uint64_t t = (diff * w) % Q;

      d_A[pos] = sum;
      d_A[pos + len] = t;
    }

    __syncthreads();
  }

  // Normalizar resultado por N^-1
  for (uint32_t i = tid; i < n; i += blockDim.x)
    d_A[i] = (d_A[i] * n_inv) % Q;
}

int main() {
  uint64_t A[N_DEF];

  /*   ------------------ Init ----------------- */

  // Rellenar A con los datos de entrada
  for (uint32_t i = 0; i < N_DEF; i++)
    A[i] = i;

  // Imprimir entrada de ejemplo
  imprimir_arreglo(A, N_DEF);

  // Precomputar twiddles
  uint64_t twiddles[MAX_TWIDDLES];
  uint64_t twiddles_inv[MAX_TWIDDLES];
  uint32_t stages = 0;
  uint32_t tmp = N_DEF;
  while (tmp > 1) {
    stages++;
    tmp >>= 1;
  }
  uint32_t *stage_offsets = (uint32_t *)malloc(sizeof(uint32_t) * stages);
  precomputar_y_copiar_twiddles(stage_offsets, twiddles, twiddles_inv);

  // Pasar los datos a memoria global del GPU
  uint64_t *d_A;
  uint64_t *d_twiddles, *d_twiddles_inv;
  uint32_t *d_stage_offsets;

  cudaMalloc(&d_A, sizeof(uint64_t) * N_DEF);
  cudaMalloc(&d_twiddles, sizeof(uint64_t) * MAX_TWIDDLES);
  cudaMalloc(&d_twiddles_inv, sizeof(uint64_t) * MAX_TWIDDLES);
  cudaMalloc(&d_stage_offsets, sizeof(uint32_t) * stages);

  cudaMemcpy(d_A, A, sizeof(uint64_t) * N_DEF, cudaMemcpyHostToDevice);
  cudaMemcpy(d_twiddles, twiddles, sizeof(uint64_t) * MAX_TWIDDLES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_twiddles_inv, twiddles_inv, sizeof(uint64_t) * MAX_TWIDDLES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_stage_offsets, stage_offsets, sizeof(uint32_t) * stages,
             cudaMemcpyHostToDevice);

  // Medicion temporal
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  /*   ------------------ NTT ----------------- */

  // Realizar NTT
  ntt_kernel<<<1, N_DEF / 2>>>(d_A, d_twiddles, d_stage_offsets);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copiar los datos de salida del GPU al Host
  cudaMemcpy(A, d_A, sizeof(uint64_t) * N_DEF, cudaMemcpyDeviceToHost);

  // Imprimir resultado de la NTT
  imprimir_arreglo(A, N_DEF);

  /*   ------------------ INTT ----------------- */

  // Realizar INTT
  uint32_t N_inv = power(N_DEF, Q - 2, Q);
  intt_kernel<<<1, N_DEF / 2>>>(d_A, d_twiddles_inv, d_stage_offsets, N_inv);

  // Copiar los datos de salida del GPU al Host
  cudaMemcpy(A, d_A, sizeof(uint64_t) * N_DEF, cudaMemcpyDeviceToHost);

  // Imprimir resultado de la INTT
  imprimir_arreglo(A, N_DEF);

  /*   ------------------ Mediciones ----------------- */

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("La NTT tardó: %f ms\n", ms);

  cudaFree(d_A);
  free(stage_offsets);
}
