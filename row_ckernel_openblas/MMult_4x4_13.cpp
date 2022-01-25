#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

void print_matrix( int m, int n, float *a, int lda )
{
  #define A( i, j ) a[ (i)*lda + (j) ]

  int i, j;

  for ( i=0; i<m; i++ ){
      for ( j=0; j<n; j++ ) {
        printf("%.1f\t", A( i,j ) );
      }
    printf("\n");
  }
  printf("\n");
}

/* Create macros so that the matrices are stored in row-major order */
#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]
#define min(i, j) ((i) < (j) ? (i): (j))
/**
About GEMM_K or kc:
1. mc = kc, since we have to maxmize (2 * mc * kc/(2 * mc + kc))
2. The equation exists provided kc << n.
3. mc * kc <= K

About GEMM_M or mc:
1. The larger mc * nc, the better calculation efficiency
2. We prepare to load A into L2 cache. Avoiding TLB miss (which would
stall CPU), subset of A should remains so until no longer needed.

About KENEL_4x4, mr=4 and nr=4
In order to move data efficiently to the registers.
Here we use C_block = A_panel x Transpose(B_panel)

In accordance to page.14 "6. MOE DETAILS YET",

L1d cahce = 32K, and L2 cache = 2MB. `getconf -a | grep PAGESIZE` = 4096.
Thus L1d is not the Cannikin, it is constraint to page size.

min_nn * kc <= PAGESIZE/2,  4 <= min_nn <= 12, so that 170 <= kc <= 512, we use 256.
After reading 6.4, rk3399 L2 cache is large, mc = 1MB / 256 = 4096 
*/
#define GEMM_N (448)  // GEMM_R
#define GEMM_M (74432)  // GEMM_P
#define GEMM_K (448)  // GEMM_Q
#define GEMM_UNROLL (4)
#define KERNEL_4x4  kernel_4x4_v1

/* Routine for computing C = A * B + C */
void packB_4(int k, int n, float* from, int ldb, float* to);
void packA_4(int m, int k, float* from, int lda, float* to);
void kernel_4x4_v1(int m, int n, int k, float* sa, float* sb, float* sc, int ldc);

/* Routine for computing C = A * B + C */

float* fastMalloc(int size)
{
    void* ptr = 0;
    int iRet = posix_memalign(&ptr, 64, size * sizeof(float));
    assert(0 == iRet);
    return (float*)ptr;
}

/* Suppose that m%4==0 and n%4==0 and k%4==0, avoiding process boundary !! */
void MY_MMult_13(int m, int n, int k, float * __restrict__ a, int lda,
                                      float * __restrict__ b, int ldb,
                                      float * __restrict__ c, int ldc )
{
    float* __restrict__ sa = fastMalloc(m * k);
    float* __restrict__ sb = fastMalloc(k * n);

    int ms, mms, ns, ks;
    int min_m, min_mm, min_n, min_k;
    for (ms = 0; ms < m; ms += GEMM_M)
	{
        min_m = m - ms;
        if (min_m > GEMM_M)
		{
            min_m = GEMM_M;
        }

        for (ks = 0; ks < k; ks += min_k)
		{
            min_k = k - ks;
            if (min_k >= (GEMM_K << 1))
			{
                min_k = GEMM_K;
            } 
			else if (min_k > GEMM_K)
			{
                min_k = (min_k / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            }

            // first packB
            min_n = n;
            if (n >= GEMM_N * 2)
			{
                min_n = GEMM_N;
            } 
			else if(n > GEMM_N)
			{
                min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
            }
            packB_4(min_k, min_n, b + ks * ldb, ldb, sb);

            // micro kernel, split A Block to smaller Panel
            for (mms = ms; mms < ms + min_m; mms += min_mm)
			{
                min_mm = (ms + min_m) - mms;
                if (min_mm >= 6 * GEMM_UNROLL)
				{
                    min_mm = 6 * GEMM_UNROLL;
                } 
				else if(min_mm >= 3 * GEMM_UNROLL) 
				{
                    min_mm = 3 * GEMM_UNROLL;
                }
				else if(min_mm > GEMM_UNROLL)
				{
                    min_mm = GEMM_UNROLL;
                }
                // coninueous packA
                packA_4(min_mm, min_k, a + mms * lda + ks, lda, sa + min_k * (mms - ms));
                KERNEL_4x4(min_mm, min_n, min_k, sa + min_k * (mms - ms), sb, c + mms * ldc, ldc);
            }

            // the first B Block has been packed, proc the others 
            for (ns = min_n; ns < n; ns += min_n)
			{
                min_n = n - ns;
                if (min_n >= GEMM_N * 2)
				{
                    min_n = GEMM_N; 
                } else if(min_n > GEMM_N)
				{
                    min_n = (min_n / 2 + GEMM_UNROLL - 1) & ~(GEMM_UNROLL - 1);
                }
                packB_4(min_k, min_n, b + ns + ldb * ks, ldb, sb);
                KERNEL_4x4(min_m, min_n, min_k, sa, sb, c + ms * ldc + ns, ldc);
            }
        }
    }

    free(sa);
    free(sb);
}


/**
pack A means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag

Output:
0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
4 4 4 4 5 5 5 5 6 6 6 6 7 7 7 7
8 8 8 8 9 9 9 9 a a a a b b b b 
c c c c d d d d e e e e f f f f

Draw it with a line
*/
void packA_4(int m, int k, float* from, int lda, float* to) {
#ifdef DEBUG_PACK_SHAPE
    printf("\n packA_4, m=%d, k=%d", m, k);
#endif
    assert( k != 0 && m != 0 && k % 4 == 0 && m % 4 == 0);
    int i, j;

    float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    float *b_offset;
    float  ctemp1,  ctemp2,  ctemp3,  ctemp4;
    float  ctemp5,  ctemp6,  ctemp7,  ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = from;
    b_offset = to;

    j = (m >> 2);
    do{
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + lda;
        a_offset3  = a_offset2 + lda;
        a_offset4  = a_offset3 + lda;
        a_offset += 4 * lda;

        i = (k >> 2);
        do{
            ctemp1  = *(a_offset1 + 0);
            ctemp2  = *(a_offset1 + 1);
            ctemp3  = *(a_offset1 + 2);
            ctemp4  = *(a_offset1 + 3);

            ctemp5  = *(a_offset2 + 0);
            ctemp6  = *(a_offset2 + 1);
            ctemp7  = *(a_offset2 + 2);
            ctemp8  = *(a_offset2 + 3);

            ctemp9  = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);

            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            *(b_offset +  0) = ctemp1;
            *(b_offset +  1) = ctemp5;
            *(b_offset +  2) = ctemp9;
            *(b_offset +  3) = ctemp13;

            *(b_offset +  4) = ctemp2;
            *(b_offset +  5) = ctemp6;
            *(b_offset +  6) = ctemp10;
            *(b_offset +  7) = ctemp14;

            *(b_offset +  8) = ctemp3;
            *(b_offset +  9) = ctemp7;
            *(b_offset + 10) = ctemp11;
            *(b_offset + 11) = ctemp15;

            *(b_offset + 12) = ctemp4;
            *(b_offset + 13) = ctemp8;
            *(b_offset + 14) = ctemp12;
            *(b_offset + 15) = ctemp16;

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            b_offset += 16;
            i --;
        }while(i > 0);
        j --;
    }while(j > 0);
}

/*
suppose that k and n is mutiple of 4
pack B means

Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7

8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f

Pack it zigzag, not like pack A

Output:
0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
8 9 a b 8 9 a b 8 9 a b 8 9 a b
4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
c d e f c d e f c d e f c d e f
*/
void packB_4(int k, int n, float* from, int ldb, float* to) {
    assert( k != 0 && n != 0 && k % 4 == 0 && n % 4 == 0);
#ifdef DEBUG_PACK_SHAPE
    printf("\n packB_4, k=%d, n=%d", k, n);
#endif

    int i, j;

    float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    float *b_offset, *b_offset1;
    float  ctemp1,  ctemp2,  ctemp3,  ctemp4;
    float  ctemp5,  ctemp6,  ctemp7,  ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;
    a_offset   = from;
    b_offset   = to;

    j = (k >> 2);
    do{
        a_offset1  = a_offset;
        a_offset2  = a_offset1 + ldb;
        a_offset3  = a_offset2 + ldb;
        a_offset4  = a_offset3 + ldb;
        a_offset  += 4 * ldb;

        b_offset1  = b_offset;
        b_offset  += 16;

        i = (n >> 2);
        do{
            ctemp1  = *(a_offset1 + 0);
            ctemp2  = *(a_offset1 + 1);
            ctemp3  = *(a_offset1 + 2);
            ctemp4  = *(a_offset1 + 3);

            ctemp5  = *(a_offset2 + 0);
            ctemp6  = *(a_offset2 + 1);
            ctemp7  = *(a_offset2 + 2);
            ctemp8  = *(a_offset2 + 3);

            ctemp9  = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);

            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            *(b_offset1 +  0) = ctemp1;
            *(b_offset1 +  1) = ctemp2;
            *(b_offset1 +  2) = ctemp3;
            *(b_offset1 +  3) = ctemp4;

            *(b_offset1 +  4) = ctemp5;
            *(b_offset1 +  5) = ctemp6;
            *(b_offset1 +  6) = ctemp7;
            *(b_offset1 +  7) = ctemp8;

            *(b_offset1 +  8) = ctemp9;
            *(b_offset1 +  9) = ctemp10;
            *(b_offset1 + 10) = ctemp11;
            *(b_offset1 + 11) = ctemp12;

            *(b_offset1 + 12) = ctemp13;
            *(b_offset1 + 13) = ctemp14;
            *(b_offset1 + 14) = ctemp15;
            *(b_offset1 + 15) = ctemp16;

            b_offset1 += k * 4;
            i --;
        }while(i > 0);
        j --;
    }while(j > 0);
}


void kernel_4x4_v1(int m, int n, int k,
    float* sa, float * sb, float* sc, int ldc) {
    assert(m > 0 && n > 0 && k > 0);
    assert(m % 4 == 0 && n % 4 == 0 && k % 4 == 0);

    float *__restrict__ a = sa, *__restrict__ b = sb, *__restrict__ c = sc;
    int i, j, l;
    for(i = 0; i < m; i += 4) {
        for(j = 0; j < n; j += 4) {
            __builtin_prefetch(b, 0, 3);
            __builtin_prefetch(a, 0, 3);

			float c_00_reg = 0;
			float c_01_reg = 0;
			float c_02_reg = 0;
			float c_03_reg = 0;
						  
			float c_10_reg = 0;
			float c_11_reg = 0;
			float c_12_reg = 0;
			float c_13_reg = 0;
						  
			float c_20_reg = 0;
			float c_21_reg = 0;
			float c_22_reg = 0;
			float c_23_reg = 0;
						  
			float c_30_reg = 0;
			float c_31_reg = 0;
			float c_32_reg = 0;
			float c_33_reg = 0;
           
            for(l = 0; l < k; l += 4)
			{
				
				for (int z = 0; z < 4; z++)
				{
					/*
					C(4x4) += A(4x1) * B(1x4) 
					        A
					---- 
					|0 |  1  2   3
					|4 |  5  6   7
					|8 |  9  10  11
					|12|  13 14  15
					----
					        B
					--------------- 
				    |0   1  2   3 |
					---------------
					 4   5  6   7
					 8   9  10  11
					 12  13 14  15

						    C
				    0   0   0    0 
					0   4   8    12
					0   8   16   24
					0   12  24   36
					*/

					float* a_00_reg = a + 0;
					float* a_10_reg = a + 1;
					float* a_20_reg = a + 2;
					float* a_30_reg = a + 3;

					float* b_00_reg = b + 0;
					float* b_01_reg = b + 1;
					float* b_02_reg = b + 2;
					float* b_03_reg = b + 3;				
				         	
					c_00_reg += *a_00_reg * *b_00_reg ;
					c_10_reg += *a_10_reg * *b_00_reg ;
					c_20_reg += *a_20_reg * *b_00_reg ;
					c_30_reg += *a_30_reg * *b_00_reg ;
                                         
					c_01_reg += *a_00_reg * *b_01_reg;
					c_11_reg += *a_10_reg * *b_01_reg;
					c_21_reg += *a_20_reg * *b_01_reg;
					c_31_reg += *a_30_reg * *b_01_reg;
                                         
					c_02_reg += *a_00_reg * *b_02_reg;
					c_12_reg += *a_10_reg * *b_02_reg;
					c_22_reg += *a_20_reg * *b_02_reg;
					c_32_reg += *a_30_reg * *b_02_reg;
                                         
					c_03_reg += *a_00_reg * *b_03_reg;
					c_13_reg += *a_10_reg * *b_03_reg;
					c_23_reg += *a_20_reg * *b_03_reg;
					c_33_reg += *a_30_reg * *b_03_reg;
					
					__builtin_prefetch(b + 4, 0, 3);
					__builtin_prefetch(a + 4, 0, 3);
					b += 4;
					a += 4;
				}
            } // endl
         
			float *c00 = c + 0 * ldc + 0;
			float *c10 = c + 1 * ldc + 0;
			float *c20 = c + 2 * ldc + 0;
			float *c30 = c + 3 * ldc + 0;
			                          
			float *c01 = c + 0 * ldc + 1;
			float *c11 = c + 1 * ldc + 1;
			float *c21 = c + 2 * ldc + 1;
			float *c31 = c + 3 * ldc + 1;
				                      
			float *c02 = c + 0 * ldc + 2;
			float *c12 = c + 1 * ldc + 2;
			float *c22 = c + 2 * ldc + 2;
			float *c32 = c + 3 * ldc + 2;
				                      
			float *c03 = c + 0 * ldc + 3;
			float *c13 = c + 1 * ldc + 3;
			float *c23 = c + 2 * ldc + 3;
			float *c33 = c + 3 * ldc + 3;

			*c00 += c_00_reg;
			*c01 += c_01_reg;
			*c02 += c_02_reg;
			*c03 += c_03_reg;

			*c10 += c_10_reg;
			*c11 += c_11_reg;
			*c12 += c_12_reg;
			*c13 += c_13_reg;

			*c20 += c_20_reg;
			*c21 += c_21_reg;
			*c22 += c_22_reg;
			*c23 += c_23_reg;
					
			*c30 += c_30_reg;
			*c31 += c_31_reg;
			*c32 += c_32_reg;
			*c33 += c_33_reg;

            c += 4;
            a -= 4*k;
        } // endj
        sc += ldc*4;
        c = sc;;
        a += 4*k;
        b = sb;
    }// endi
}
