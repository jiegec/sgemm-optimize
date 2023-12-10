#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/arm/neon.h"
#include "simde/arm/neon/mla_lane.h"

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define SMALL_BLOCK_SIZE 4

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_generic(int lda, int M, int N, int K, float *restrict A, float *restrict B, float *restrict C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      float cij = C[i + j * lda];
      for (int k = 0; k < K; ++k)
        cij += A[i + k * lda] * B[k + j * lda];
      C[i + j * lda] = cij;
    }
}

// let compiler optimize for M = N = SMALL_BLOCK_SIZE
// so that numbers can reside in registers
// lda, ldb, ldc: load stripe
// A: SMALL_BLOCK_SIZE * K
// B: K * SMALL_BLOCK_SIZE
// C: SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE
static void do_block_small(int K, int lda, float *restrict A, int ldb, float *restrict B, int ldc, float *restrict C)
{
  int M = SMALL_BLOCK_SIZE, N = SMALL_BLOCK_SIZE;
  // four columns of C
  float32x4_t C0, C1, C2, C3;
  // temporaries
  float32x4_t a, b;
  // local buffer to numbers in C
  float CC[SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE];
  for (int j = 0; j < SMALL_BLOCK_SIZE; j++)
  {
    for (int i = 0; i < SMALL_BLOCK_SIZE; i++)
    {
      CC[j + i * SMALL_BLOCK_SIZE] = C[i + j * ldc];
    }
  }

  // pack
  C0 = vld1q_f32(CC + 0 * SMALL_BLOCK_SIZE);
  C1 = vld1q_f32(CC + 1 * SMALL_BLOCK_SIZE);
  C2 = vld1q_f32(CC + 2 * SMALL_BLOCK_SIZE);
  C3 = vld1q_f32(CC + 3 * SMALL_BLOCK_SIZE);

  for (int k = 0; k < K; ++k)
  {
    /* Compute C(i,j) */
    b = vld1q_f32(B + k * ldb);
    a = vld1q_dup_f32(A + 0 + k * lda);
    C0 = vmlaq_f32(C0, a, b);

    a = vld1q_dup_f32(A + 1 + k * lda);
    C1 = vmlaq_f32(C1, a, b);

    a = vld1q_dup_f32(A + 2 + k * lda);
    C2 = vmlaq_f32(C2, a, b);

    a = vld1q_dup_f32(A + 3 + k * lda);
    C3 = vmlaq_f32(C3, a, b);
  }

  // unpack
  vst1q_f32(CC + 0 * SMALL_BLOCK_SIZE, C0);
  vst1q_f32(CC + 1 * SMALL_BLOCK_SIZE, C1);
  vst1q_f32(CC + 2 * SMALL_BLOCK_SIZE, C2);
  vst1q_f32(CC + 3 * SMALL_BLOCK_SIZE, C3);
  for (int j = 0; j < SMALL_BLOCK_SIZE; ++j)
  {
    for (int i = 0; i < SMALL_BLOCK_SIZE; ++i)
    {
      C[i + j * ldc] = CC[j + i * SMALL_BLOCK_SIZE];
    }
  }
}

// two level blocking
// A: MxK, B: KxN, C: MxN
static void do_block_large(int M, int N, int K, int lda, float *restrict A, int ldb, float *restrict B, int ldc, float *restrict C)
{
  // buffer for packing
  float AA[BLOCK_SIZE * BLOCK_SIZE];
  float BB[BLOCK_SIZE * BLOCK_SIZE];

  /* For each block-column of C */
  for (int j = 0; j < N; j += SMALL_BLOCK_SIZE)
  {
    int NN = min(SMALL_BLOCK_SIZE, N - j);
    // pack B with transpose
    for (int ii = 0; ii < K; ii++)
    {
      for (int jj = 0; jj < SMALL_BLOCK_SIZE; jj++)
      {
        BB[jj + ii * SMALL_BLOCK_SIZE] = B[ii + (jj + j) * lda];
      }
    }

    /* For each block-row of C */
    for (int i = 0; i < M; i += SMALL_BLOCK_SIZE)
    {
      int MM = min(SMALL_BLOCK_SIZE, M - i);

      // pack A only once
      if (j == 0)
      {
        for (int jj = 0; jj < K; jj++)
        {
          for (int ii = 0; ii < SMALL_BLOCK_SIZE; ii++)
          {
            AA[ii + jj * SMALL_BLOCK_SIZE + i * K] = A[(ii + i) + jj * lda];
          }
        }
      }

      if (MM == SMALL_BLOCK_SIZE && NN == SMALL_BLOCK_SIZE)
      {
        /* Perform individual block sgemm */

        do_block_small(K, SMALL_BLOCK_SIZE, AA + i * K, SMALL_BLOCK_SIZE, BB, lda, C + i + j * lda);
      }
      else
      {
        /* Perform individual block sgemm */
        do_block_generic(lda, MM, NN, K, A + i + 0 * lda, B + 0 + j * lda, C + i + j * lda);
      }
    }
  }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_sgemm(int lda, float *restrict A, float *restrict B, float *restrict C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE)
  {
    int M = min(BLOCK_SIZE, lda - i);
    /* For each block-column of A */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
      int K = min(BLOCK_SIZE, lda - j);

      do_block_large(M, lda, K, lda, A + i + j * lda, lda, B + j + 0 * lda, lda, C + i + 0 * lda);
    }
  }
}
