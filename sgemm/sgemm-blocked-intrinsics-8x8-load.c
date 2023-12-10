#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/arm/neon.h"
#include "simde/arm/neon/mla_lane.h"

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define SMALL_BLOCK_SIZE 8

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
  // 16 registers
  // C00: C[0, 0-3], C04: C[0, 4-7]
  float32x4_t C00, C04, C10, C14, C20, C24, C30, C34, C40, C44, C50, C54, C60, C64, C70, C74;
  // temporaries
  float32x4_t b0, b4;
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
  C00 = vld1q_f32(CC + 0 * SMALL_BLOCK_SIZE + 0);
  C04 = vld1q_f32(CC + 0 * SMALL_BLOCK_SIZE + 4);
  C10 = vld1q_f32(CC + 1 * SMALL_BLOCK_SIZE + 0);
  C14 = vld1q_f32(CC + 1 * SMALL_BLOCK_SIZE + 4);
  C20 = vld1q_f32(CC + 2 * SMALL_BLOCK_SIZE + 0);
  C24 = vld1q_f32(CC + 2 * SMALL_BLOCK_SIZE + 4);
  C30 = vld1q_f32(CC + 3 * SMALL_BLOCK_SIZE + 0);
  C34 = vld1q_f32(CC + 3 * SMALL_BLOCK_SIZE + 4);
  C40 = vld1q_f32(CC + 4 * SMALL_BLOCK_SIZE + 0);
  C44 = vld1q_f32(CC + 4 * SMALL_BLOCK_SIZE + 4);
  C50 = vld1q_f32(CC + 5 * SMALL_BLOCK_SIZE + 0);
  C54 = vld1q_f32(CC + 5 * SMALL_BLOCK_SIZE + 4);
  C60 = vld1q_f32(CC + 6 * SMALL_BLOCK_SIZE + 0);
  C64 = vld1q_f32(CC + 6 * SMALL_BLOCK_SIZE + 4);
  C70 = vld1q_f32(CC + 7 * SMALL_BLOCK_SIZE + 0);
  C74 = vld1q_f32(CC + 7 * SMALL_BLOCK_SIZE + 4);

#pragma GCC unroll 8
  for (int k = 0; k < K; ++k)
  {
    //__builtin_prefetch(A + (k + 1) * lda);
    /* Compute C(i,j) */
    b0 = vld1q_f32(B + k * ldb + 0);
    b4 = vld1q_f32(B + k * ldb + 4);

    float32x4_t A0 = vld1q_f32(A + k * lda);
    C00 = vmlaq_laneq_f32(C00, b0, A0, 0);
    C04 = vmlaq_laneq_f32(C04, b4, A0, 0);
    C10 = vmlaq_laneq_f32(C10, b0, A0, 1);
    C14 = vmlaq_laneq_f32(C14, b4, A0, 1);
    C20 = vmlaq_laneq_f32(C20, b0, A0, 2);
    C24 = vmlaq_laneq_f32(C24, b4, A0, 2);
    C30 = vmlaq_laneq_f32(C30, b0, A0, 3);
    C34 = vmlaq_laneq_f32(C34, b4, A0, 3);

    float32x4_t A4 = vld1q_f32(A + 4 + k * lda);
    C40 = vmlaq_laneq_f32(C40, b0, A4, 0);
    C44 = vmlaq_laneq_f32(C44, b4, A4, 0);
    C50 = vmlaq_laneq_f32(C50, b0, A4, 1);
    C54 = vmlaq_laneq_f32(C54, b4, A4, 1);
    C60 = vmlaq_laneq_f32(C60, b0, A4, 2);
    C64 = vmlaq_laneq_f32(C64, b4, A4, 2);
    C70 = vmlaq_laneq_f32(C70, b0, A4, 3);
    C74 = vmlaq_laneq_f32(C74, b4, A4, 3);
  }

  // unpack
  vst1q_f32(CC + 0 * SMALL_BLOCK_SIZE + 0, C00);
  vst1q_f32(CC + 0 * SMALL_BLOCK_SIZE + 4, C04);
  vst1q_f32(CC + 1 * SMALL_BLOCK_SIZE + 0, C10);
  vst1q_f32(CC + 1 * SMALL_BLOCK_SIZE + 4, C14);
  vst1q_f32(CC + 2 * SMALL_BLOCK_SIZE + 0, C20);
  vst1q_f32(CC + 2 * SMALL_BLOCK_SIZE + 4, C24);
  vst1q_f32(CC + 3 * SMALL_BLOCK_SIZE + 0, C30);
  vst1q_f32(CC + 3 * SMALL_BLOCK_SIZE + 4, C34);
  vst1q_f32(CC + 4 * SMALL_BLOCK_SIZE + 0, C40);
  vst1q_f32(CC + 4 * SMALL_BLOCK_SIZE + 4, C44);
  vst1q_f32(CC + 5 * SMALL_BLOCK_SIZE + 0, C50);
  vst1q_f32(CC + 5 * SMALL_BLOCK_SIZE + 4, C54);
  vst1q_f32(CC + 6 * SMALL_BLOCK_SIZE + 0, C60);
  vst1q_f32(CC + 6 * SMALL_BLOCK_SIZE + 4, C64);
  vst1q_f32(CC + 7 * SMALL_BLOCK_SIZE + 0, C70);
  vst1q_f32(CC + 7 * SMALL_BLOCK_SIZE + 4, C74);
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
