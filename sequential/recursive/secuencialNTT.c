#include <bits/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define Q_DEF	3329
#define N_DEF	8
#define W_POS	749

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

/// <summary> 
/// Realiza una NTT (Number Theoretic Transform) con convolución cíclica positiva.
/// </summary>
///	
/// <param name="A">Puntero al arreglo donde se almacenará el resultado de la NTT.</param>
/// <param name="a">Puntero al arreglo de coeficientes de entrada.</param>
/// <param name="N">Tamaño de los arreglos (debe ser potencia de 2 para algoritmos optimizados).</param>
/// <param name="w">Raíz primitiva de orden N módulo q.</param>
/// <param name="q">Número entero primo que define el cuerpo finito.</param>
void NTT_ciclica_secuencial(long* A, int* a, int N, int w, int q) {
	for (int k = 0; k < N; k++) {
		A[k] = 0;
		for (int n = 0; n < N; n++) {
			long termino = (a[n] * power(w, n*k, q)) % q; 
			A[k] = (A[k] + termino) % q;
		}
	}
}

void imprimir_arreglo(void* arreglo, size_t size, int tamanio) {
    for (int i = 0; i < tamanio; i++) {
        if (size == sizeof(int)) {
            printf("%d ", ((int*)arreglo)[i]);
        } 
        else if (size == sizeof(long)) {
            printf("%ld ", ((long*)arreglo)[i]);
        }
    }
    printf("\n");
}

int main(int argc, char* argv[]) {

	long A[N_DEF];
	int a[N_DEF];

	for (int i = 0; i < N_DEF; i++) {
		a[i] = i;
	}

	imprimir_arreglo((void*)a, sizeof(int), N_DEF);
	puts("");

	struct timespec inicio, fin;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &inicio);

	NTT_ciclica_secuencial(A, a, N_DEF, W_POS, Q_DEF);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &fin);

	double nanosegundos = fin.tv_nsec - inicio.tv_nsec;

	imprimir_arreglo((void*)A, sizeof(long), N_DEF);
	puts("");

	printf("El algoritmo secuencial NTT tardó: %.2lf ns\n", nanosegundos);


	return 0;
}
