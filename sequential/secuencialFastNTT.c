#include <bits/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define Q_DEF	17
#define N_DEF	8
#define W_POS	9

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
/// Realiza una Fast-NTT.
/// </summary>
///	
/// <param name="A">Puntero al arreglo donde se almacenará el resultado de la NTT.</param>
/// <param name="a">Puntero al arreglo de coeficientes de entrada.</param>
/// <param name="N">Tamaño de los arreglos (debe ser potencia de 2 para algoritmos optimizados).</param>
/// <param name="w">Raíz primitiva de orden N módulo q.</param>
/// <param name="q">Número entero primo que define el cuerpo finito.</param>
void FastNTT_ciclica_secuencial(long* A, int* a, int N, int w, int q) {
	
	if (N == 1) {
		A[0] = a[0];
		return;
	}

	long* E = (long*)malloc(sizeof(long) * N/2);
	long* O = (long*)malloc(sizeof(long) * N/2);
	
	// int pares[N/2];
	// int impares[N/2];
	int* pares = (int*)malloc(sizeof(int) * N/2);
	int* impares = (int*)malloc(sizeof(int) * N/2);
	
	for (int i = 0; i < N/2;  i++) {
		pares[i] = a[2*i];
		impares[i] = a[2*i+1];
	}
	
	long w2 = power(w, 2, q); // Raiz para tamaño N/2
	FastNTT_ciclica_secuencial(E, pares, N/2, w2, q);
	FastNTT_ciclica_secuencial(O, impares, N/2, w2, q);

	long twiddle = 1;
	long wstep = w;
	for (int k = 0; k < N/2; k++) {

		long t = (twiddle  * O[k]) % q;
		A[k] = (E[k] + t) % q;

		long temp = (E[k] - t) % q;
		if (temp < 0) temp += q;
		A[k+N/2] = temp;

		twiddle = (twiddle * wstep) % q;
	}

	free(pares);
	free(impares);
	free(O);
	free(E);
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

	for (int i = 0; i < N_DEF/2; i++) {
		a[i] = i;
		a[N_DEF-i-1] = i;
	}

	imprimir_arreglo((void*)a, sizeof(int), N_DEF);

	struct timespec inicio, fin;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &inicio);

	// Fast-NTT ciclica
	FastNTT_ciclica_secuencial(A, a, N_DEF, W_POS, Q_DEF);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &fin);

	double nanosegundos = fin.tv_nsec - inicio.tv_nsec;

	imprimir_arreglo((void*)A, sizeof(long), N_DEF);

	printf("El algoritmo secuencial Fast-NTT tardó: %.2lf ns\n", nanosegundos);

	return 0;
}
