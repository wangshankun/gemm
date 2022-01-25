#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <cstring>
#include "cblas.h"

using namespace std;

void matrix_a(float*buf, int m, int n)
{
    int total = m * n;
    for (int i = 0; i < total; i++ )
    {
       buf[i]= (float)i;
    }
}
void matrix_b(float*buf, int m, int n)
{
    int total = m * n;
    for (int i = 0; i < total; i++ )
    {
       buf[i]= (float)(total - i);
    }
}


//格式与numpy一致i是列的index; j是行的index
//不实现转置操作，因此忽略lda,ldb,ldc参数
#define A(j, i) a[ (j)*k + (i) ]
#define B(j, i) b[ (j)*n + (i) ]
#define C(j, i) c[ (j)*n + (i) ]
void native_c(int m, int n, int k, float *a, float *b, float *c)
{
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int p = 0; p < k; p++)
            {
                C(j, i) += A(j, p) * B(p, i);
            }
        }
    }
}

//断开指令依赖，求c值的时候并行做，而不是一个做完再做下一个
void wean_c(int m, int n, int k, float *a, float *b, float *c)
{
    for (int j = 0; j < m; j++)
    {
        for (int p = 0; p < k; p++)
        {
            for (int i = 0; i < n; i++)
            {
                C(j, i) += A(j, p) * B(p, i);
            }
        }
    }
}

void random_matrix(float*buf, int len)
{
    int i = 0;
    srand48(time(0));
    for ( i = 0; i < len; i++ )
    {
       buf[i]= 2.0 * (float)drand48() - 1.0;
    }
}

void matmult(int m, int n, int k, float *a, int lda,
                                  float *b, int ldb,
                                  float *c, int ldc );
int main()
{
    //int m = 10240, n = 10240, k = 10240;
    //int m = 1024, n = 1024, k = 1024;
    //int m = 64, n = 64, k = 64;
    int m = 300, n = 300, k = 300;
    int lda = k, ldb = n, ldc = n;

    int sampling_index = (int)(m * n * 0.771);
    float* abuff = (float*)malloc(m  * k * 4);
    float* bbuff = (float*)malloc(n  * k * 4);
    float* cbuff = (float*)malloc(m  * n * 4);

    random_matrix(abuff, m*k);
    //matrix_a(abuff, m, k);
    //matrix_a(bbuff, m, k);
    //matrix_b(bbuff, k, n);
    random_matrix(bbuff, k*n);
    memset(cbuff, 0 , m * n * sizeof(float));

    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, abuff, lda, bbuff, ldb, 0.0, cbuff, ldc);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("openblas elapsed  time:%f cbuff[%d]:%f\r\n",elapsed, sampling_index, cbuff[sampling_index]);
    
    //printf("==tatal:===m:%d n:%d ldc:%d ======\r\n", m, n, ldc);
    //print_matrix(m, n, cbuff,      ldc);

    memset(cbuff, 0 , m * n * sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &start);
    matmult(m, n, k, abuff, lda, bbuff, ldb, cbuff, ldc);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("matmult elapsed  time:%f cbuff[%d]:%f\r\n",elapsed, sampling_index, cbuff[sampling_index]);

    free(abuff);
    free(bbuff);
    free(cbuff);
    return 0;
}

