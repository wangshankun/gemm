#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "cblas.h"
#include <string.h>

inline void* pacage_a_row(int m, int k, float* from, int lda, float* to)
{
	for(int i = 0; i < m; i++)
	{
		memcpy(to + i * k, from + lda * i, sizeof(float) * k);
	}
}

inline void* pacage_b_row(int k, int n, float* from, int ldb, float* to)
{
	for(int i = 0; i < k; i++)
	{
		memcpy(to + i * n, from + ldb * i, sizeof(float) * n);
	}
}

float* fastMalloc(int size)
{
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 64, size * sizeof(float));
    assert(0 == iRet);
    return ptr;
}


void matmult(int m, int n, int k, float* a, int lda,
                                  float* b, int ldb,
                                  float* c, int ldc )
{
    memset(c, 0, n*k*sizeof(float));//默认c是0，如果传入beta为1,删除这行
    int ms, ns, ks;
    int min_m, min_n, min_k;
    #define GEMM_M 1
    #define GEMM_N 64
    #define GEMM_K 64

	float* sa = fastMalloc(GEMM_M * GEMM_K);
	float* sb = fastMalloc(GEMM_K * GEMM_N);

    for (ms = 0; ms < m; ms += min_m)
    {
        min_m = GEMM_M;

        for (ks = 0; ks < k; ks += min_k)
        {
            min_k = k - ks;
            if (min_k >= GEMM_K)
            {
                min_k = GEMM_K;
            }
			
			pacage_a_row(min_m, min_k, a + ms * lda + ks, m, sa);
            for (ns = 0; ns < n; ns += min_n)
            {
                min_n = n - ns;
                if (min_n >= GEMM_N)
                {
                    min_n = GEMM_N; 
                }
                //kernel_64(min_m, min_n, min_k, a + ms * lda + ks, b + ks * ldb, c + ms * ldc + ns, ldc);
				pacage_b_row(min_k, min_n, b + ns + ldb * ks, ldb, sb);

                int min_lda = min_k;
                int min_ldb = min_n;
                int min_ldc = min_n;
                /*
                //if( min_n == GEMM_N && min_k == GEMM_K )
                //{
                //    //光计算
                //}
                
                if(min_k < GEMM_K)
                {
                    //a后面补0，继续光计算
                    for(int i = min_k; i < GEMM_K; i++)
                    {
                        sa[i] = 0;
                    }
                }

                if(min_n < GEMM_N)
                {
                    //只取c的前min_n个结果
                }
                */
                //用openblas做对比测试
                //printf("%d  %d %d\r\n",min_m, min_n, min_k);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                             min_m, min_n, min_k, 
                             1.0,
                             sa, min_lda,
                             sb, min_ldb, 
                             1.0, //因为k分块后c的累计,这里beta不能是0
                             c + ms * ldc + ns, min_ldc);
            }
        }
    }
	
	free(sa);
	free(sb);
}