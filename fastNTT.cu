// NTT 2D para multiplicación de polinomios bivariables A(x,y) * B(x,y)
// Estrategia: NTT separable por filas y columnas (Cooley-Tukey 2D)
//
// Layout de memoria: row-major, A[row][col] = A[row * NX + col]
// Flujo:
//   NTT_filas(A), NTT_filas(B)
//   Transponer(A), Transponer(B)
//   NTT_filas(A^T), NTT_filas(B^T)   <- esto equivale a NTT_columnas
//   Multiplicación puntual: C = A * B
//   INTT_filas(C)
//   Transponer(C)
//   INTT_filas(C^T)                   <- INTT_columnas
//   Normalizar(C)

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CEILING(NUM,DEN) ((NUM + DEN - 1)/DEN)

// ============================================================
//  Configuración: dimensiones y primo
// ============================================================

// NX y NY deben ser potencias de 2
// El primo Q debe satisfacer Q-1 divisible por max(NX, NY)
#define NX 2048
#define NY 2048
#define Q 998244353U // primo NTT-friendly: 998244353 = 119 * 2^23 + 1
#define G 3U         // raíz primitiva de Q

// ============================================================
//  Aritmética de Montgomery
// ============================================================

#define R_BITS 32
#define R (1ULL << R_BITS)

static uint32_t compute_n_inv(uint32_t n) {
  uint32_t x = 1;
  for (int i = 0; i < 31; i++)
    x *= 2 - n * x;
  return -x;
}

__host__ __device__ __forceinline__ uint32_t mont_reduce(uint64_t x,
                                                         uint32_t n_inv) {
  uint32_t m = (uint32_t)x * n_inv;
  uint32_t t = (x + (uint64_t)m * Q) >> 32;
  return t >= Q ? t - Q : t;
}

__host__ __device__ __forceinline__ uint32_t mont_mul(uint32_t a, uint32_t b,
                                                      uint32_t n_inv) {
  return mont_reduce((uint64_t)a * b, n_inv);
}

// ============================================================
//  Utilidades en host
// ============================================================

static uint32_t power(uint32_t base, uint32_t exp, uint32_t mod) {
  uint64_t res = 1;
  base %= mod;
  while (exp > 0) {
    if (exp & 1)
      res = res * base % mod;
    base = (uint64_t)base * base % mod;
    exp >>= 1;
  }
  return (uint32_t)res;
}

static void print_array(const char *label, uint32_t *A, uint32_t rows,
                        uint32_t cols) {
  printf("%s (%ux%u):\n", label, rows, cols);
  uint32_t show_r = rows < 4 ? rows : 4;
  uint32_t show_c = cols < 8 ? cols : 8;
  for (uint32_t r = 0; r < show_r; r++) {
    for (uint32_t c = 0; c < show_c; c++)
      printf("%10u ", A[r * cols + c]);
    printf("...\n");
  }
  printf("...\n\n");
}

// ============================================================
//  Precomputo de twiddles para una longitud N dada
// ============================================================

static void precompute_twiddles(uint32_t N, uint32_t *stage_offsets,
                                uint32_t *twiddles, uint32_t *twiddles_inv,
                                uint32_t R2, uint32_t n_inv) {
  uint32_t w = power(G, (Q - 1) / N, Q);
  uint32_t w_inv = power(w, Q - 2, Q);

  uint32_t offset = 0, stage = 0;
  for (uint32_t len = 2; len <= N; len <<= 1) {
    stage_offsets[stage] = offset;
    uint32_t mid = len >> 1;
    for (uint32_t k = 0; k < mid; k++) {
      uint32_t exp = (N / len) * k;
      twiddles[offset + k] = mont_mul(power(w, exp, Q), R2, n_inv);
      twiddles_inv[offset + k] = mont_mul(power(w_inv, exp, Q), R2, n_inv);
    }
    offset += mid;
    stage++;
  }
}

// ============================================================
//  KERNELS
// ============================================================

// Convierte N elementos a representación Montgomery
__global__ void to_montgomery(uint32_t *d_A, uint32_t R2, uint32_t n_inv,
                              uint32_t N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  d_A[tid] = mont_mul(d_A[tid], R2, n_inv);
}

// ---- NTT block (primeros 9 stages, shared memory 512 elementos) ----
__global__ void ntt_block(uint32_t *d_A, const uint32_t *twiddles,
                          const uint32_t *stage_offsets, uint32_t n_inv,
                          uint32_t total_stages) {
  __shared__ uint32_t s_A[512];

  uint32_t tid = threadIdx.x;
  uint32_t base = blockIdx.x * (blockDim.x << 1);

  s_A[tid] = d_A[base + tid];
  s_A[tid + blockDim.x] = d_A[base + tid + blockDim.x];
  __syncthreads();

  uint32_t stage = 0;

  // Warp stages (no necesitan __syncthreads)
  for (; stage < 5 && stage < total_stages; stage++) {
    uint32_t len = 1u << stage;
    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos], v = s_A[pos + len];
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

  // Intra-block stages
  for (; stage < 9 && stage < total_stages; stage++) {
    uint32_t len = 1u << stage;
    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos], v = s_A[pos + len];
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

  d_A[base + tid] = s_A[tid];
  d_A[base + tid + blockDim.x] = s_A[tid + blockDim.x];
}

// ---- NTT stage individual (stages >= 9) ----
__global__ void ntt_stage(uint32_t *d_A, const uint32_t *twiddles,
                          const uint32_t *stage_offsets, uint32_t stage,
                          uint32_t len, uint32_t n_inv, uint32_t N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N / 2)
    return;

  uint32_t j = tid & (len - 1);
  uint32_t group = tid >> stage;
  uint32_t pos = (group << (stage + 1)) + j;

  uint32_t u = d_A[pos], v = d_A[pos + len];
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

// ---- INTT block ----
__global__ void intt_block(uint32_t *d_A, const uint32_t *twiddles_inv,
                           const uint32_t *stage_offsets, uint32_t n_inv,
                           uint32_t total_stages) {
  __shared__ uint32_t s_A[512];

  uint32_t tid = threadIdx.x;
  uint32_t base = blockIdx.x * (blockDim.x << 1);

  s_A[tid] = d_A[base + tid];
  s_A[tid + blockDim.x] = d_A[base + tid + blockDim.x];
  __syncthreads();

  // Intra-block (stages grandes primero en INTT)
  uint32_t intra = total_stages < 9 ? total_stages : 9;
  for (uint32_t ls = 0; ls < 3 && ls < intra; ls++) {
    uint32_t stage = (intra - 1) - ls;
    uint32_t len = 1u << stage;
    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos], v = s_A[pos + len];
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

  // Warp stages (pequeños)
  for (uint32_t ls = 3; ls < intra; ls++) {
    uint32_t stage = (intra - 1) - ls;
    uint32_t len = 1u << stage;
    uint32_t j = tid & (len - 1);
    uint32_t group = tid >> stage;
    uint32_t pos = (group << (stage + 1)) + j;

    uint32_t u = s_A[pos], v = s_A[pos + len];
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

  d_A[base + tid] = s_A[tid];
  d_A[base + tid + blockDim.x] = s_A[tid + blockDim.x];
}

// ---- INTT stage individual ----
__global__ void intt_stage(uint32_t *d_A, const uint32_t *twiddles_inv,
                           const uint32_t *stage_offsets, uint32_t stage,
                           uint32_t len, uint32_t n_inv, uint32_t N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N / 2)
    return;

  uint32_t j = tid & (len - 1);
  uint32_t group = tid >> stage;
  uint32_t pos = (group << (stage + 1)) + j;

  uint32_t u = d_A[pos], v = d_A[pos + len];
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

// ---- Normalización final INTT ----
__global__ void intt_normalize(uint32_t *d_A, uint32_t n_inv,
                               uint32_t N_inv_mont, uint32_t N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N)
    return;
  uint32_t val = mont_reduce((uint64_t)d_A[tid], n_inv);
  d_A[tid] = mont_mul(val, N_inv_mont, n_inv);
}

// ---- Multiplicación puntual en dominio frecuencial ----
__global__ void pointwise_mul(uint32_t *d_C, const uint32_t *d_A,
                              const uint32_t *d_B, uint32_t n_inv,
                              uint32_t total) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;
  d_C[tid] = mont_mul(d_A[tid], d_B[tid], n_inv);
}

// ---- Transposición con shared memory (evita bank conflicts) ----

#define TILE 32

__global__ void transpose(uint32_t *out, const uint32_t *in, uint32_t rows,
                          uint32_t cols) {
  __shared__ uint32_t tile[TILE][TILE + 1]; // +1 evita bank conflicts

  uint32_t x = blockIdx.x * TILE + threadIdx.x; // col origen
  uint32_t y = blockIdx.y * TILE + threadIdx.y; // fila origen

  if (x < cols && y < rows)
    tile[threadIdx.y][threadIdx.x] = in[y * cols + x];

  __syncthreads();

  // Escribir transpuesto: la columna x pasa a ser fila, fila y pasa a columna
  x = blockIdx.y * TILE + threadIdx.x; // col destino
  y = blockIdx.x * TILE + threadIdx.y; // fila destino

  if (x < rows && y < cols)
    out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

// ============================================================
//  Estructuras de configuración NTT para una dimensión
// ============================================================

struct NTTConfig {
  uint32_t N;
  uint32_t total_stages; // log2(N)
  uint32_t n_inv;        // -Q^{-1} mod 2^32
  uint32_t R2;           // R^2 mod Q (en forma estándar)
  uint32_t *d_twiddles;
  uint32_t *d_twiddles_inv;
  uint32_t *d_stage_offsets;
};

static NTTConfig make_ntt_config(uint32_t N) {
  NTTConfig cfg;
  cfg.N = N;

  cfg.total_stages = 0;
  for (uint32_t tmp = N; tmp > 1; tmp >>= 1)
    cfg.total_stages++;

  cfg.n_inv = compute_n_inv(Q);
  cfg.R2 = (uint32_t)((uint64_t)(R % Q) * (R % Q) % Q);

  uint32_t max_twiddles = N - 1;
  uint32_t *h_twiddles = (uint32_t *)malloc(sizeof(uint32_t) * max_twiddles);
  uint32_t *h_twiddles_inv =
      (uint32_t *)malloc(sizeof(uint32_t) * max_twiddles);
  uint32_t *h_offsets = (uint32_t *)malloc(sizeof(uint32_t) * cfg.total_stages);

  precompute_twiddles(N, h_offsets, h_twiddles, h_twiddles_inv, cfg.R2,
                      cfg.n_inv);

  cudaMalloc(&cfg.d_twiddles, sizeof(uint32_t) * max_twiddles);
  cudaMalloc(&cfg.d_twiddles_inv, sizeof(uint32_t) * max_twiddles);
  cudaMalloc(&cfg.d_stage_offsets, sizeof(uint32_t) * cfg.total_stages);

  cudaMemcpy(cfg.d_twiddles, h_twiddles, sizeof(uint32_t) * max_twiddles,
             cudaMemcpyHostToDevice);
  cudaMemcpy(cfg.d_twiddles_inv, h_twiddles_inv,
             sizeof(uint32_t) * max_twiddles, cudaMemcpyHostToDevice);
  cudaMemcpy(cfg.d_stage_offsets, h_offsets,
             sizeof(uint32_t) * cfg.total_stages, cudaMemcpyHostToDevice);

  free(h_twiddles);
  free(h_twiddles_inv);
  free(h_offsets);
  return cfg;
}

static void free_ntt_config(NTTConfig &cfg) {
  cudaFree(cfg.d_twiddles);
  cudaFree(cfg.d_twiddles_inv);
  cudaFree(cfg.d_stage_offsets);
}

// ============================================================
//  Bit-reversal permutation en GPU
//  Cada thread intercambia el elemento tid con su bit-reversed j (si j > tid).
//  blockIdx.y selecciona la fila dentro del batch.
// ============================================================

__global__ void bit_reversal(uint32_t *d_A, uint32_t N, uint32_t log2N) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t base = blockIdx.y * N;

  if (tid >= N)
    return;

  // Calcular índice bit-reversed de tid
  uint32_t j = 0, x = tid;
  for (uint32_t b = 0; b < log2N; b++) {
    j = (j << 1) | (x & 1);
    x >>= 1;
  }

  // Intercambiar solo una vez
  if (j > tid) {
    uint32_t tmp = d_A[base + tid];
    d_A[base + tid] = d_A[base + j];
    d_A[base + j] = tmp;
  }
}

static void apply_bit_reversal(uint32_t *d_A, uint32_t N, uint32_t log2N,
                               uint32_t count) {
  const uint32_t threads = 256;
  uint32_t ceilingXBlocks = CEILING(N, threads);// (N + threads - 1) / threads; // N/256
  dim3 blks(ceilingXBlocks, count);
  bit_reversal<<<blks, threads>>>(d_A, N, log2N);
}

// ============================================================
//  Batch NTT: aplica NTT a `count` arreglos contiguos de longitud N
//  Cada arreglo empieza en d_A + i * N
// ============================================================

static void batch_ntt(uint32_t *d_A, const NTTConfig &cfg, uint32_t count,
                      bool forward) {
  const uint32_t N = cfg.N;
  const uint32_t block_size = N >= 512 ? 256 : N / 2;
  const uint32_t elems_per_blk = block_size * 2;
  const uint32_t blks_per_row = CEILING(N, elems_per_blk); // (N + elems_per_blk - 1) / elems_per_blk;
  const uint32_t stage_blks = (N / 2 + 255) / 256;
  const uint32_t blk_stage_lim = N >= 512 ? 9u : cfg.total_stages;
  const uint32_t first_ext_len = N >= 512 ? 512u : N;

  if (forward) {
    // Bit-reversal sobre todas las filas del batch de una sola vez
    apply_bit_reversal(d_A, N, cfg.total_stages, count);

    for (uint32_t i = 0; i < count; i++) {
      uint32_t *ptr = d_A + i * N;
      ntt_block<<<blks_per_row, block_size>>>(ptr, cfg.d_twiddles,
                                              cfg.d_stage_offsets, cfg.n_inv,
                                              cfg.total_stages);

      uint32_t stage = blk_stage_lim;
      for (uint32_t len = first_ext_len; len < N; len <<= 1, stage++)
        ntt_stage<<<stage_blks, 256>>>(ptr, cfg.d_twiddles, cfg.d_stage_offsets,
                                       stage, len, cfg.n_inv, N);
    }
  } else {
    for (uint32_t i = 0; i < count; i++) {
      uint32_t *ptr = d_A + i * N;
      uint32_t stage = cfg.total_stages - 1;
      for (uint32_t len = N >> 1; len >= first_ext_len; len >>= 1, stage--)
        intt_stage<<<stage_blks, 256>>>(ptr, cfg.d_twiddles_inv,
                                        cfg.d_stage_offsets, stage, len,
                                        cfg.n_inv, N);

      intt_block<<<blks_per_row, block_size>>>(ptr, cfg.d_twiddles_inv,
                                               cfg.d_stage_offsets, cfg.n_inv,
                                               cfg.total_stages);
    }
    // Bit-reversal al final de la INTT (deshace el reordenamiento)
    apply_bit_reversal(d_A, N, cfg.total_stages, count);
  }
}

// ============================================================
//  NTT 2D: filas → transponer → filas (= columnas) y viceversa
// ============================================================

static void ntt2d(uint32_t *d_A, uint32_t *d_tmp, const NTTConfig &cfg_x,
                  const NTTConfig &cfg_y, bool forward) {

  dim3 tile_threads(TILE, TILE);

  if (forward) {
    // 1. Convertir a Montgomery
    uint32_t total = NX * NY;
    uint32_t blks = CEILING(total,256); // (total + 255) / 256;
    to_montgomery<<<blks, 256>>>(d_A, cfg_x.R2, cfg_x.n_inv, total);

    // 2. NTT por filas: NY filas de longitud NX
    //    Después: d_A[ky * NX + kx_index]  (aun en layout filas)
    batch_ntt(d_A, cfg_x, NY, true);

    // 3. Transponer [NY x NX] → [NX x NY]
    //    Ahora la fila i corresponde a la columna original i
    dim3 blks_t(CEILING(NX,TILE), CEILING(NY,TILE));
    transpose<<<blks_t, tile_threads>>>(d_tmp, d_A, NY, NX);
    cudaMemcpy(d_A, d_tmp, sizeof(uint32_t) * total,
               cudaMemcpyDeviceToDevice);

    // 4. NTT por columnas: NX "filas" (que son columnas originales) de longitud
    // NY
    //    Al terminar d_A queda en layout transpuesto [NX x NY]:
    //    d_A[kx * NY + ky]
    batch_ntt(d_A, cfg_y, NX, true);

  } else {
    // INTT: d_A llega en layout [NX x NY] = d_A[kx * NY + ky]

    // 1. INTT por columnas (son filas en el layout actual)
    batch_ntt(d_A, cfg_y, NX, false);

    // 2. Transponer [NX x NY] → [NY x NX] para recuperar layout de filas
    dim3 blks_t(CEILING(NY,TILE), CEILING(NX,TILE));
    transpose<<<blks_t, tile_threads>>>(d_tmp, d_A, NX, NY);
    cudaMemcpy(d_A, d_tmp, sizeof(uint32_t) * NX * NY,
               cudaMemcpyDeviceToDevice);

    // 3. INTT por filas
    batch_ntt(d_A, cfg_x, NY, false);

    // 4. Normalizar
    uint32_t NXY = NX * NY;
    uint32_t N_inv = power(NXY % Q, Q - 2, Q);
    uint32_t n_inv = cfg_x.n_inv;
    uint32_t R2 = cfg_x.R2;
    uint32_t N_inv_mont = mont_mul(N_inv, R2, n_inv);

    uint32_t blks = CEILING(NXY, 256); // (NXY + 255) / 256;
    intt_normalize<<<blks, 256>>>(d_A, n_inv, N_inv_mont, NXY);
  }
}

// ============================================================
//  Multiplicación de polinomios bivariables
//  A(x,y) * B(x,y) mod (x^NX - 1, y^NY - 1)
// ============================================================

void poly2d_mul(uint32_t *d_A, uint32_t *d_B, uint32_t *d_C, uint32_t *d_tmp,
                const NTTConfig &cfg_x, const NTTConfig &cfg_y) {

  uint32_t total = NX * NY;
  uint32_t n_inv = cfg_x.n_inv;

  // Transformar A y B al dominio frecuencial 2D
  ntt2d(d_A, d_tmp, cfg_x, cfg_y, true);
  ntt2d(d_B, d_tmp, cfg_x, cfg_y, true);

  // Multiplicación puntual: C = A * B (ambos ya en Montgomery)
  uint32_t blks =  CEILING(total, 256); //(total + 255) / 256;
  pointwise_mul<<<blks, 256>>>(d_C, d_A, d_B, n_inv, total);

  // Transformar C de vuelta al dominio polinomial
  ntt2d(d_C, d_tmp, cfg_x, cfg_y, false);
}

// ============================================================
//  Main de demostración
// ============================================================

int main_demo() {
  printf("NTT 2D: multiplicación de polinomios bivariables\n");
  printf("Dimensiones: NX=%u, NY=%u, Q=%u\n\n", NX, NY, Q);

  // Verificar que Q-1 sea divisible por NX y NY (condición NTT)
  if ((Q - 1) % NX != 0 || (Q - 1) % NY != 0) {
    printf("Error: Q-1 no es divisible por NX o NY.\n");
    return 1;
  }

  uint32_t total = NX * NY;

  // Inicializar polinomios de prueba en host
  uint32_t *h_A = (uint32_t *)malloc(sizeof(uint32_t) * total);
  uint32_t *h_B = (uint32_t *)malloc(sizeof(uint32_t) * total);
  uint32_t *h_C = (uint32_t *)malloc(sizeof(uint32_t) * total);

  for (uint32_t i = 0; i < total; i++) {
    h_A[i] = i % Q;
    h_B[i] = (total - i) % Q;
  }

  print_array("A (entrada)", h_A, NY, NX);
  print_array("B (entrada)", h_B, NY, NX);

  // Alojar memoria en GPU
  uint32_t *d_A, *d_B, *d_C, *d_tmp;
  cudaMalloc(&d_A, sizeof(uint32_t) * total);
  cudaMalloc(&d_B, sizeof(uint32_t) * total);
  cudaMalloc(&d_C, sizeof(uint32_t) * total);
  cudaMalloc(&d_tmp, sizeof(uint32_t) * total); // buffer para transponer

  cudaMemcpy(d_A, h_A, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);

  // Precomputar configuraciones NTT para cada dimensión
  NTTConfig cfg_x = make_ntt_config(NX);
  NTTConfig cfg_y = make_ntt_config(NY);

  // Medición de tiempo
  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);
  cudaEventRecord(ev_start);

  // ---- Multiplicación ----
  poly2d_mul(d_A, d_B, d_C, d_tmp, cfg_x, cfg_y);

  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  // Copiar resultado al host
  cudaMemcpy(h_C, d_C, sizeof(uint32_t) * total, cudaMemcpyDeviceToHost);
  print_array("C = A*B (resultado)", h_C, NY, NX);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  printf("Tiempo total (NTT2D + mul + INTT2D): %.3f ms\n", ms);

  // Liberar recursos
  free_ntt_config(cfg_x);
  free_ntt_config(cfg_y);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_tmp);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  return 0;
}
