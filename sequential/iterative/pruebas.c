#include <time.h> 
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define Q  12289

double diff_us(struct timespec a, struct timespec b) {
    return (b.tv_sec  - a.tv_sec)  * 1e6 +
           (b.tv_nsec - a.tv_nsec) / 1e3;
}

long power(long base, long exp, int mod) {
    long res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

void generar_datos(int* a, int N, int q) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % q;
    }
}

void NTT_ciclica_secuencial(long* A, int* a, int N, int w, int q) {
	for (int k = 0; k < N; k++) {
		A[k] = 0;
		for (int n = 0; n < N; n++) {
			long termino = (a[n] * power(w, n*k, q)) % q; 
			A[k] = (A[k] + termino) % q;
		}
	}
}

void FastNTT_ciclica_iterativa(uint64_t* A, uint32_t* a, uint32_t N, uint32_t w, uint32_t q)
{
    // Copiar entrada
    for (uint32_t i = 0; i < N; i++)
        A[i] = a[i];

    // --- Bit-reversal permutation ---
    uint32_t j = 0;
    for (uint32_t i = 1; i < N; i++) {
        uint32_t bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;

        if (i < j) {
            uint64_t temp = A[i];
            A[i] = A[j];
            A[j] = temp;
        }
    }

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

int comparar(long* A, long* B, int N) {
    for (int i = 0; i < N; i++) {
        if (A[i] != B[i]) {
            printf("❌ Error en indice %d: Fast=%ld  Directa=%ld\n",
                   i, A[i], B[i]);
            return 0;
        }
    }
    return 1;
}

// Para un funcionamiento correcto:
// 1. N es potencia de 2
// 2. N divide a (q-1)
// 3. w es raiz positiva de orden N mod q
int main() {

    srand(time(NULL));

    int Ns[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(Ns) / sizeof(Ns[0]);
    int pruebas = 10;

    for (int s = 0; s < num_sizes; s++) {

        int N = Ns[s];
		int g = 11;
		int w = power(g, (Q-1)/N, Q);

        int*  a      = malloc(sizeof(int)  * N);
        long* A_fast = malloc(sizeof(long) * N);
        long* A_ref  = malloc(sizeof(long) * N);

        double tiempo_fast = 0.0;
        double tiempo_ref  = 0.0;

        for (int t = 0; t < pruebas; t++) {

            generar_datos(a, N, Q);

            struct timespec t0, t1;

            clock_gettime(CLOCK_MONOTONIC, &t0);
            FastNTT_ciclica_iterativa(A_fast, a, N, w, Q);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            tiempo_fast += diff_us(t0, t1);

            clock_gettime(CLOCK_MONOTONIC, &t0);
            NTT_ciclica_secuencial(A_ref, a, N, w, Q);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            tiempo_ref += diff_us(t0, t1);

            if (!comparar(A_fast, A_ref, N)) {
                printf("❌ Error detectado para N = %d\n", N);
                return 1;
            }
        }

        printf("\nN = %d\n", N);
        printf("FastNTT   promedio: %.0lf us\n", tiempo_fast / pruebas);
        printf("NTT lenta promedio: %.0lf us\n", tiempo_ref  / pruebas);
        printf("Speedup: %.2fx\n",
               tiempo_ref / tiempo_fast);

        free(a);
        free(A_fast);
        free(A_ref);
    }

    return 0;
}
