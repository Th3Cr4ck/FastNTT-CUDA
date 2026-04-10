// ntt2d_test.cu
// Verificación y benchmark para la NTT 2D
//
// Tests:
//   TEST 1 — round-trip:       INTT(NTT(A)) == A
//   TEST 2 — modelo de oro 1D: NTT_GPU(fila) == NTT_CPU_secuencial(fila)
//   TEST 3 — convolucion 2D:   poly2d_mul_GPU == convolucion_ciclica_CPU
//
// Benchmark:
//   Mide tiempo promedio de poly2d_mul en GPU y CPU para tamanios
//   NxN con N = 16, 32, 64, 128, 256, 512, 1024, 2048, ...

#include "fastNTT.cu"
#include <time.h>

// ============================================================
//  Dimensiones para los tests (pequeñas para CPU)
// ============================================================

#define TEST_NX 16
#define TEST_NY 16

// Numero de repeticiones para el benchmark
#define BENCH_REPS 5

// ============================================================
//  Utilidades comunes
// ============================================================

static bool arrays_equal(const uint32_t *a, const uint32_t *b, uint32_t n,
                         uint32_t max_print) {
  uint32_t errors = 0;
  for (uint32_t i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      if (errors < max_print)
        printf("  [%u] esperado=%u  obtenido=%u\n", i, a[i], b[i]);
      errors++;
    }
  }
  if (errors > max_print)
    printf("  ... y %u diferencias mas\n", errors - max_print);
  return errors == 0;
}

static void random_poly(uint32_t *A, uint32_t n, uint32_t seed) {
  uint32_t s = seed;
  for (uint32_t i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;
    A[i] = s % Q;
  }
}

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ============================================================
//  ntt2d_param / poly2d_mul_param
//  Versiones de ntt2d y poly2d_mul parametrizadas por Nx, Ny
//  en lugar de usar los #define NX/NY.
//  Los tests y el benchmark las usan para trabajar con tamanios variables.
// ============================================================

static void ntt2d_param(uint32_t *d_A, uint32_t *d_tmp, const NTTConfig *cfg_x,
                        const NTTConfig *cfg_y, uint32_t Nx, uint32_t Ny,
                        bool forward) {
  dim3 tile_t(TILE, TILE);
  uint32_t total = Nx * Ny;

  if (forward) {
    {
      uint32_t blks = (total + 255) / 256;
      to_montgomery<<<blks, 256>>>(d_A, cfg_x->R2, cfg_x->n_inv, total);
    }
    batch_ntt(d_A, *cfg_x, Ny, true);
    {
      dim3 b((Nx + TILE - 1) / TILE, (Ny + TILE - 1) / TILE);
      transpose<<<b, tile_t>>>(d_tmp, d_A, Ny, Nx);
      cudaMemcpy(d_A, d_tmp, sizeof(uint32_t) * total,
                 cudaMemcpyDeviceToDevice);
    }
    batch_ntt(d_A, *cfg_y, Nx, true);
  } else {
    batch_ntt(d_A, *cfg_y, Nx, false);
    {
      dim3 b((Ny + TILE - 1) / TILE, (Nx + TILE - 1) / TILE);
      transpose<<<b, tile_t>>>(d_tmp, d_A, Nx, Ny);
      cudaMemcpy(d_A, d_tmp, sizeof(uint32_t) * total,
                 cudaMemcpyDeviceToDevice);
    }
    batch_ntt(d_A, *cfg_x, Ny, false);
    {
      uint32_t NXY = Nx * Ny;
      uint32_t N_inv = power(NXY % Q, Q - 2, Q);
      uint32_t N_inv_mont = mont_mul(N_inv, cfg_x->R2, cfg_x->n_inv);
      uint32_t blks = (NXY + 255) / 256;
      intt_normalize<<<blks, 256>>>(d_A, cfg_x->n_inv, N_inv_mont, NXY);
    }
  }
}

static void poly2d_mul_param(uint32_t *d_A, uint32_t *d_B, uint32_t *d_C,
                             uint32_t *d_tmp, const NTTConfig *cfg_x,
                             const NTTConfig *cfg_y, uint32_t Nx, uint32_t Ny) {
  uint32_t total = Nx * Ny;
  ntt2d_param(d_A, d_tmp, cfg_x, cfg_y, Nx, Ny, true);
  ntt2d_param(d_B, d_tmp, cfg_x, cfg_y, Nx, Ny, true);
  {
    uint32_t blks = (total + 255) / 256;
    pointwise_mul<<<blks, 256>>>(d_C, d_A, d_B, cfg_x->n_inv, total);
  }
  ntt2d_param(d_C, d_tmp, cfg_x, cfg_y, Nx, Ny, false);
}

// ============================================================
//  TEST 1 — Round-trip: INTT(NTT(A)) == A
// ============================================================

bool test_roundtrip(void) {
  printf("=== TEST 1: round-trip INTT(NTT(A)) == A ===\n");

  uint32_t total = NX * NY;
  uint32_t *h_orig = (uint32_t *)malloc(sizeof(uint32_t) * total);
  uint32_t *h_res = (uint32_t *)malloc(sizeof(uint32_t) * total);
  random_poly(h_orig, total, 1234);

  uint32_t *d_A, *d_tmp;
  cudaMalloc(&d_A, sizeof(uint32_t) * total);
  cudaMalloc(&d_tmp, sizeof(uint32_t) * total);
  cudaMemcpy(d_A, h_orig, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);

  NTTConfig cfg_x = make_ntt_config(NX);
  NTTConfig cfg_y = make_ntt_config(NY);

  ntt2d(d_A, d_tmp, cfg_x, cfg_y, true);
  ntt2d(d_A, d_tmp, cfg_x, cfg_y, false);

  cudaMemcpy(h_res, d_A, sizeof(uint32_t) * total, cudaMemcpyDeviceToHost);

  bool ok = arrays_equal(h_orig, h_res, total, 8);
  printf("  Resultado: %s\n\n", ok ? "PASS" : "FAIL");

  free_ntt_config(cfg_x);
  free_ntt_config(cfg_y);
  free(h_orig);
  free(h_res);
  cudaFree(d_A);
  cudaFree(d_tmp);
  return ok;
}

// ============================================================
//  TEST 2 — Modelo de oro 1D
//  NTT_GPU(fila) == NTT_CPU_secuencial(fila)
// ============================================================

static void ntt_cpu_ref(uint32_t *A, uint32_t N) {
  // Bit-reversal permutation
  for (uint32_t i = 1, j = 0; i < N; i++) {
    uint32_t bit = N >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      uint32_t tmp = A[i];
      A[i] = A[j];
      A[j] = tmp;
    }
  }
  uint32_t w_N = power(G, (Q - 1) / N, Q);
  for (uint32_t len = 2; len <= N; len <<= 1) {
    uint32_t w_len = power(w_N, N / len, Q);
    for (uint32_t i = 0; i < N; i += len) {
      uint64_t w = 1;
      for (uint32_t j = 0; j < len / 2; j++) {
        uint32_t u = A[i + j];
        uint32_t v = (uint64_t)A[i + j + len / 2] * w % Q;
        A[i + j] = (u + v) % Q;
        A[i + j + len / 2] = (u + Q - v) % Q;
        w = w * w_len % Q;
      }
    }
  }
}

bool test_ntt1d_vs_reference(void) {
  printf("=== TEST 2: NTT_GPU (1 fila) == NTT_CPU_referencia ===\n");

  uint32_t *h_gpu = (uint32_t *)malloc(sizeof(uint32_t) * NX);
  uint32_t *h_ref = (uint32_t *)malloc(sizeof(uint32_t) * NX);
  random_poly(h_gpu, NX, 9999);
  memcpy(h_ref, h_gpu, sizeof(uint32_t) * NX);

  ntt_cpu_ref(h_ref, NX);

  NTTConfig cfg = make_ntt_config(NX);
  uint32_t *d_row;
  cudaMalloc(&d_row, sizeof(uint32_t) * NX);
  cudaMemcpy(d_row, h_gpu, sizeof(uint32_t) * NX, cudaMemcpyHostToDevice);

  {
    uint32_t blks = (NX + 255) / 256;
    to_montgomery<<<blks, 256>>>(d_row, cfg.R2, cfg.n_inv, NX);
  }
  batch_ntt(d_row, cfg, 1, true);
  {
    // Salir de Montgomery sin escalar (N_inv = 1)
    uint32_t N_inv_mont = mont_mul(1u, cfg.R2, cfg.n_inv);
    intt_normalize<<<(NX + 255) / 256, 256>>>(d_row, cfg.n_inv, N_inv_mont, NX);
  }

  uint32_t *h_gpu_out = (uint32_t *)malloc(sizeof(uint32_t) * NX);
  cudaMemcpy(h_gpu_out, d_row, sizeof(uint32_t) * NX, cudaMemcpyDeviceToHost);

  bool ok = arrays_equal(h_ref, h_gpu_out, NX, 8);
  printf("  Resultado: %s\n\n", ok ? "PASS" : "FAIL");

  free_ntt_config(cfg);
  free(h_gpu);
  free(h_ref);
  free(h_gpu_out);
  cudaFree(d_row);
  return ok;
}

// ============================================================
//  TEST 3 — Convolucion 2D: GPU == referencia directa O(N^4)
// ============================================================

static void conv2d_direct_ref(const uint32_t *A, const uint32_t *B, uint32_t *C,
                              uint32_t Nx, uint32_t Ny) {
  for (uint32_t i = 0; i < Ny; i++) {
    for (uint32_t j = 0; j < Nx; j++) {
      uint64_t acc = 0;
      for (uint32_t r = 0; r < Ny; r++) {
        for (uint32_t s = 0; s < Nx; s++) {
          uint32_t bi = (i + Ny - r) % Ny;
          uint32_t bj = (j + Nx - s) % Nx;
          acc += (uint64_t)A[r * Nx + s] * B[bi * Nx + bj] % Q;
          acc %= Q;
        }
      }
      C[i * Nx + j] = (uint32_t)acc;
    }
  }
}

bool test_conv2d(void) {
  printf("=== TEST 3: poly2d_mul_GPU == convolucion_ciclica_CPU ===\n");
  printf("  (dimensiones %ux%u)\n", TEST_NY, TEST_NX);

  if ((Q - 1) % TEST_NX != 0 || (Q - 1) % TEST_NY != 0) {
    printf("  SKIP: Q-1 no divisible por TEST_NX o TEST_NY\n\n");
    return true;
  }

  uint32_t total = TEST_NX * TEST_NY;
  uint32_t *h_A = (uint32_t *)malloc(sizeof(uint32_t) * total);
  uint32_t *h_B = (uint32_t *)malloc(sizeof(uint32_t) * total);
  uint32_t *h_C = (uint32_t *)malloc(sizeof(uint32_t) * total);
  uint32_t *h_ref = (uint32_t *)malloc(sizeof(uint32_t) * total);

  random_poly(h_A, total, 111);
  random_poly(h_B, total, 222);

  printf("  Calculando referencia CPU directa O(N^4)...\n");
  conv2d_direct_ref(h_A, h_B, h_ref, TEST_NX, TEST_NY);

  uint32_t *d_A, *d_B, *d_C, *d_tmp;
  cudaMalloc(&d_A, sizeof(uint32_t) * total);
  cudaMalloc(&d_B, sizeof(uint32_t) * total);
  cudaMalloc(&d_C, sizeof(uint32_t) * total);
  cudaMalloc(&d_tmp, sizeof(uint32_t) * total);

  cudaMemcpy(d_A, h_A, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);

  NTTConfig cfg_x = make_ntt_config(TEST_NX);
  NTTConfig cfg_y = make_ntt_config(TEST_NY);

  poly2d_mul_param(d_A, d_B, d_C, d_tmp, &cfg_x, &cfg_y, TEST_NX, TEST_NY);

  cudaMemcpy(h_C, d_C, sizeof(uint32_t) * total, cudaMemcpyDeviceToHost);

  bool ok = arrays_equal(h_ref, h_C, total, 8);
  printf("  Resultado: %s\n\n", ok ? "PASS" : "FAIL");

  free_ntt_config(cfg_x);
  free_ntt_config(cfg_y);
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_tmp);
  return ok;
}

// ============================================================
//  Implementacion CPU con NTT: modelo de oro para el benchmark
//
//  ntt_cpu_ref  — NTT 1D iterativa con bit-reversal
//  intt_cpu_ref — INTT 1D iterativa con bit-reversal
//  ntt2d_cpu    — NTT 2D separable: filas luego columnas
//  poly2d_mul_cpu — multiplicacion de polinomios bivariables en CPU
// ============================================================

static void intt_cpu_ref(uint32_t *A, uint32_t N) {
  uint32_t w_inv = power(power(G, (Q - 1) / N, Q), Q - 2, Q);

  // Bit-reversal
  for (uint32_t i = 1, j = 0; i < N; i++) {
    uint32_t bit = N >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j) {
      uint32_t tmp = A[i];
      A[i] = A[j];
      A[j] = tmp;
    }
  }

  for (uint32_t len = 2; len <= N; len <<= 1) {
    uint32_t w_len = power(w_inv, N / len, Q);
    for (uint32_t i = 0; i < N; i += len) {
      uint64_t w = 1;
      for (uint32_t j = 0; j < len / 2; j++) {
        uint32_t u = A[i + j];
        uint32_t v = (uint64_t)A[i + j + len / 2] * w % Q;
        A[i + j] = (u + v) % Q;
        A[i + j + len / 2] = (u + Q - v) % Q;
        w = w * w_len % Q;
      }
    }
  }

  uint32_t N_inv = power(N, Q - 2, Q);
  for (uint32_t i = 0; i < N; i++)
    A[i] = (uint64_t)A[i] * N_inv % Q;
}

static void ntt2d_cpu(uint32_t *A, uint32_t Nx, uint32_t Ny, bool forward) {
  uint32_t *col = (uint32_t *)malloc(sizeof(uint32_t) * Ny);

  for (uint32_t r = 0; r < Ny; r++) {
    if (forward)
      ntt_cpu_ref(A + r * Nx, Nx);
    else
      intt_cpu_ref(A + r * Nx, Nx);
  }

  for (uint32_t c = 0; c < Nx; c++) {
    for (uint32_t r = 0; r < Ny; r++)
      col[r] = A[r * Nx + c];
    if (forward)
      ntt_cpu_ref(col, Ny);
    else
      intt_cpu_ref(col, Ny);
    for (uint32_t r = 0; r < Ny; r++)
      A[r * Nx + c] = col[r];
  }

  free(col);
}

static void poly2d_mul_cpu(const uint32_t *A, const uint32_t *B, uint32_t *C,
                           uint32_t Nx, uint32_t Ny) {
  uint32_t total = Nx * Ny;
  uint32_t *fA = (uint32_t *)malloc(sizeof(uint32_t) * total);
  uint32_t *fB = (uint32_t *)malloc(sizeof(uint32_t) * total);

  memcpy(fA, A, sizeof(uint32_t) * total);
  memcpy(fB, B, sizeof(uint32_t) * total);

  ntt2d_cpu(fA, Nx, Ny, true);
  ntt2d_cpu(fB, Nx, Ny, true);

  for (uint32_t i = 0; i < total; i++)
    C[i] = (uint64_t)fA[i] * fB[i] % Q;

  ntt2d_cpu(C, Nx, Ny, false);

  free(fA);
  free(fB);
}

// ============================================================
//  BENCHMARK — GPU vs CPU, tamanios NxN crecientes
// ============================================================

void benchmark(void) {
  static const uint32_t sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2048};
  static const uint32_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);

  printf("================================================================\n");
  printf("  BENCHMARK: poly2d_mul  (promedio %d repeticiones)\n", BENCH_REPS);
  printf("  Q = %u\n", Q);
  printf("================================================================\n");
  printf("  %-6s  %-14s  %-14s  %-10s\n", "N", "GPU (ms)", "CPU (ms)",
         "Speedup");
  printf("  %-6s  %-14s  %-14s  %-10s\n", "------", "--------------",
         "--------------", "----------");

  for (uint32_t si = 0; si < n_sizes; si++) {
    uint32_t N = sizes[si];
    uint32_t total = N * N;

    if ((Q - 1) % N != 0) {
      printf("  %-6u  SKIP (Q-1 no divisible por N)\n", N);
      continue;
    }

    uint32_t *h_A = (uint32_t *)malloc(sizeof(uint32_t) * total);
    uint32_t *h_B = (uint32_t *)malloc(sizeof(uint32_t) * total);
    uint32_t *h_C = (uint32_t *)malloc(sizeof(uint32_t) * total);
    random_poly(h_A, total, 0xABCD + si);
    random_poly(h_B, total, 0x1234 + si);

    // ---- GPU ----
    uint32_t *d_A, *d_B, *d_C, *d_tmp;
    cudaMalloc(&d_A, sizeof(uint32_t) * total);
    cudaMalloc(&d_B, sizeof(uint32_t) * total);
    cudaMalloc(&d_C, sizeof(uint32_t) * total);
    cudaMalloc(&d_tmp, sizeof(uint32_t) * total);

    NTTConfig cfg_x = make_ntt_config(N);
    NTTConfig cfg_y = make_ntt_config(N);

    // Warmup GPU
    cudaMemcpy(d_A, h_A, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);
    poly2d_mul_param(d_A, d_B, d_C, d_tmp, &cfg_x, &cfg_y, N, N);
    cudaDeviceSynchronize();

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    float gpu_total_ms = 0.0f;
    for (int r = 0; r < BENCH_REPS; r++) {
      // Recargar A y B (poly2d_mul los modifica)
      cudaMemcpy(d_A, h_A, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B, sizeof(uint32_t) * total, cudaMemcpyHostToDevice);
      cudaEventRecord(ev_start);
      poly2d_mul_param(d_A, d_B, d_C, d_tmp, &cfg_x, &cfg_y, N, N);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      gpu_total_ms += ms;
    }
    float gpu_avg_ms = gpu_total_ms / BENCH_REPS;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free_ntt_config(cfg_x);
    free_ntt_config(cfg_y);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_tmp);

    // ---- CPU ----
    // Warmup CPU
    poly2d_mul_cpu(h_A, h_B, h_C, N, N);

    double cpu_total_s = 0.0;
    for (int r = 0; r < BENCH_REPS; r++) {
      double t0 = now_sec();
      poly2d_mul_cpu(h_A, h_B, h_C, N, N);
      double t1 = now_sec();
      cpu_total_s += t1 - t0;
    }
    double cpu_avg_ms = cpu_total_s / BENCH_REPS * 1000.0;
    double speedup = cpu_avg_ms / (double)gpu_avg_ms;

    printf("  %-6u  %-14.3f  %-14.3f  %.2fx\n", N, gpu_avg_ms, cpu_avg_ms,
           speedup);

    free(h_A);
    free(h_B);
    free(h_C);
  }

  printf("================================================================\n");
  printf("  Tiempos GPU incluyen transferencias H->D de A y B.\n");
  printf(
      "================================================================\n\n");
}

// ============================================================
//  Main
// ============================================================

int main(void) {
  printf("============================================\n");
  printf("  NTT 2D -- Suite de verificacion\n");
  printf("  NX=%u  NY=%u  Q=%u\n", NX, NY, Q);
  printf("============================================\n\n");

  int passed = 0;
  int total_tests = 3;

  passed += test_roundtrip() ? 1 : 0;
  passed += test_ntt1d_vs_reference() ? 1 : 0;
  passed += test_conv2d() ? 1 : 0;

  printf("============================================\n");
  printf("  %d / %d tests pasaron\n", passed, total_tests);
  printf("============================================\n\n");

  if (passed == total_tests)
    benchmark();

  return passed == total_tests ? 0 : 1;
}
