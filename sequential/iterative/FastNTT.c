#include <bits/time.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define Q_DEF	3329
#define N_DEF	8
#define W_POS	749

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

/// <summary>
/// Realiza una Fast-NTT iterativa radix-2 (DIT).
/// </summary>
/// <param name="A">Arreglo donde se almacenará el resultado (puede ser el mismo que a).</param>
/// <param name="a">Arreglo de coeficientes de entrada.</param>
/// <param name="N">Tamaño (potencia de 2).</param>
/// <param name="w">Raíz primitiva de orden N módulo q.</param>
/// <param name="q">Primo del cuerpo finito.</param>
void FastNTT_ciclica_iterativa(uint64_t* A, uint32_t* a, uint32_t N, uint32_t w, uint32_t q)
{
    // Copiar entrada
    for (uint32_t i = 0; i < N; i++)
        A[i] = a[i];

    /* // --- Bit-reversal permutation ---
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
    } */

    // --- Etapas ---
    for (uint32_t len = 2; len <= N; len <<= 1) {

        uint64_t wlen = power(w, (N_DEF / len), q);  // raíz para esta etapa

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

void FastINTT_ciclica_iterativa(uint64_t* A,
                                uint32_t N,
                                uint32_t w,
                                uint32_t q)
{
    // Inversa de la raíz
    uint64_t w_inv = power(w, q - 2, q);

    // Etapas (DIF: empezamos desde N y vamos bajando)
    for (uint32_t len = N; len >= 2; len >>= 1) {

        uint64_t wlen = power(w_inv, N_DEF / len, q);

        for (uint32_t i = 0; i < N; i += len) {

            uint64_t twiddle = 1;

            for (uint32_t k = 0; k < len / 2; k++) {

                uint64_t u = A[i + k];
                uint64_t v = A[i + k + len/2];

                // Mariposa DIF
                uint64_t sum  = (u + v) % q;
                uint64_t diff = (u + q - v) % q;

                A[i + k] = sum;
                A[i + k + len/2] = (diff * twiddle) % q;

                twiddle = (twiddle * wlen) % q;
            }
        }

        if (len == 2) break;  // evitar underflow de uint32_t
    }

    // Multiplicar por N^{-1}
    uint64_t N_inv = power(N, q - 2, q);

    for (uint32_t i = 0; i < N; i++) {
        A[i] = (A[i] * N_inv) % q;
    }
}

void imprimir_arreglo(void* arreglo, size_t size, uint32_t tamanio) {
    for (uint32_t i = 0; i < tamanio; i++) {
        if (size == sizeof(uint32_t)) {
            printf("%d ", ((uint32_t*)arreglo)[i]);
        } 
        else if (size == sizeof(uint64_t)) {
            printf("%ld ", ((uint64_t*)arreglo)[i]);
        }
    }
    printf("\n");
}

int main(uint32_t argc, char* argv[]) {

	uint64_t A[N_DEF];
	uint32_t a[N_DEF];

	for (uint32_t i = 0; i < N_DEF; i++) {
		a[i] = i;
	}

	imprimir_arreglo((void*)a, sizeof(uint32_t), N_DEF);
	puts("");

	struct timespec inicio, fin;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &inicio);

	FastNTT_ciclica_iterativa(A, a, N_DEF, W_POS, Q_DEF);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &fin);

	double nanosegundos = fin.tv_nsec - inicio.tv_nsec;

	imprimir_arreglo((void*)A, sizeof(uint64_t), N_DEF);
	puts("");

	FastINTT_ciclica_iterativa(A, N_DEF, W_POS, Q_DEF);
	imprimir_arreglo((void*)A, sizeof(uint64_t), N_DEF);
	puts("");

	printf("El algoritmo secuencial NTT tardó: %.2lf ns\n", nanosegundos);

	return 0;
}
