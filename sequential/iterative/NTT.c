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
/// Realiza una NTT (Number Theoretic Transform) con convolución cíclica positiva.
/// </summary>
///	
/// <param name="A">Puntero al arreglo donde se almacenará el resultado de la NTT.</param>
/// <param name="a">Puntero al arreglo de coeficientes de entrada.</param>
/// <param name="N">Tamaño de los arreglos (debe ser potencia de 2 para algoritmos optimizados).</param>
/// <param name="w">Raíz primitiva de orden N módulo q.</param>
/// <param name="q">Número entero primo que define el cuerpo finito.</param>
void NTT_ciclica_secuencial(uint64_t* A, uint32_t* a, uint32_t N, uint32_t w, uint32_t q) {
	for (uint32_t k = 0; k < N; k++) {
		A[k] = 0;
		for (uint32_t n = 0; n < N; n++) {
			uint64_t termino = (a[n] * power(w, n*k, q)) % q; 
			A[k] = (A[k] + termino) % q;
		}
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

	NTT_ciclica_secuencial(A, a, N_DEF, W_POS, Q_DEF);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &fin);

	double nanosegundos = fin.tv_nsec - inicio.tv_nsec;

	imprimir_arreglo((void*)A, sizeof(uint64_t), N_DEF);
	puts("");

	printf("El algoritmo secuencial NTT tardó: %.2lf ns\n", nanosegundos);

	return 0;
}
