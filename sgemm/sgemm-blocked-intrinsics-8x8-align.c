#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/arm/neon.h"
#include "simde/arm/neon/mla_lane.h"

const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 128
#endif

#define SMALL_BLOCK_SIZE 8

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block_generic(int lda, int M, int N, int K, float *restrict A, float *restrict B, float *restrict C)
{
  for (int k = 0; k < K; ++k)
  {
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
      {
        /* Compute C(i,j) */
        float cij = C[i + j * lda];
        cij += A[i + k * lda] * B[k + j * lda];
        C[i + j * lda] = cij;
      }
    }
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
  // four rows of C
  // 16 registers
  // C00: C[0-3, 0], C40: C[4-7, 0]
  float32x4_t C00, C40, C01, C41, C02, C42, C03, C43, C04, C44, C05, C45, C06, C46, C07, C47;
  // temporaries
  float32x4_t a0, a4;

  // pack
  C00 = vld1q_f32(C + 0 * ldc + 0);
  C40 = vld1q_f32(C + 0 * ldc + 4);
  C01 = vld1q_f32(C + 1 * ldc + 0);
  C41 = vld1q_f32(C + 1 * ldc + 4);
  C02 = vld1q_f32(C + 2 * ldc + 0);
  C42 = vld1q_f32(C + 2 * ldc + 4);
  C03 = vld1q_f32(C + 3 * ldc + 0);
  C43 = vld1q_f32(C + 3 * ldc + 4);
  C04 = vld1q_f32(C + 4 * ldc + 0);
  C44 = vld1q_f32(C + 4 * ldc + 4);
  C05 = vld1q_f32(C + 5 * ldc + 0);
  C45 = vld1q_f32(C + 5 * ldc + 4);
  C06 = vld1q_f32(C + 6 * ldc + 0);
  C46 = vld1q_f32(C + 6 * ldc + 4);
  C07 = vld1q_f32(C + 7 * ldc + 0);
  C47 = vld1q_f32(C + 7 * ldc + 4);

#pragma GCC unroll 8
  for (int k = 0; k < K; ++k)
  {
    /* Compute C(i,j) */
    a0 = vld1q_f32(A + k * ldb + 0);
    a4 = vld1q_f32(A + k * ldb + 4);

    float32x4_t B0 = vld1q_f32(B + k * lda);
    C00 = vmlaq_laneq_f32(C00, a0, B0, 0);
    C40 = vmlaq_laneq_f32(C40, a4, B0, 0);
    C01 = vmlaq_laneq_f32(C01, a0, B0, 1);
    C41 = vmlaq_laneq_f32(C41, a4, B0, 1);
    C02 = vmlaq_laneq_f32(C02, a0, B0, 2);
    C42 = vmlaq_laneq_f32(C42, a4, B0, 2);
    C03 = vmlaq_laneq_f32(C03, a0, B0, 3);
    C43 = vmlaq_laneq_f32(C43, a4, B0, 3);

    float32x4_t B4 = vld1q_f32(B + 4 + k * lda);
    C04 = vmlaq_laneq_f32(C04, a0, B4, 0);
    C44 = vmlaq_laneq_f32(C44, a4, B4, 0);
    C05 = vmlaq_laneq_f32(C05, a0, B4, 1);
    C45 = vmlaq_laneq_f32(C45, a4, B4, 1);
    C06 = vmlaq_laneq_f32(C06, a0, B4, 2);
    C46 = vmlaq_laneq_f32(C46, a4, B4, 2);
    C07 = vmlaq_laneq_f32(C07, a0, B4, 3);
    C47 = vmlaq_laneq_f32(C47, a4, B4, 3);
  }

  // unpack
  vst1q_f32(C + 0 * ldc + 0, C00);
  vst1q_f32(C + 0 * ldc + 4, C40);
  vst1q_f32(C + 1 * ldc + 0, C01);
  vst1q_f32(C + 1 * ldc + 4, C41);
  vst1q_f32(C + 2 * ldc + 0, C02);
  vst1q_f32(C + 2 * ldc + 4, C42);
  vst1q_f32(C + 3 * ldc + 0, C03);
  vst1q_f32(C + 3 * ldc + 4, C43);
  vst1q_f32(C + 4 * ldc + 0, C04);
  vst1q_f32(C + 4 * ldc + 4, C44);
  vst1q_f32(C + 5 * ldc + 0, C05);
  vst1q_f32(C + 5 * ldc + 4, C45);
  vst1q_f32(C + 6 * ldc + 0, C06);
  vst1q_f32(C + 6 * ldc + 4, C46);
  vst1q_f32(C + 7 * ldc + 0, C07);
  vst1q_f32(C + 7 * ldc + 4, C47);
}

// two level blocking
// A: MxK, B: KxN, C: MxN
static void do_block_large(int M, int N, int K, int lda, float *restrict A, int ldb, float *restrict B, int ldc, float *restrict C)
{
  // buffer for packing
  float AA[BLOCK_SIZE * BLOCK_SIZE];
  float BB[BLOCK_SIZE * BLOCK_SIZE];
  float CC[SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE];

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

      /* Perform individual block sgemm */
      if (MM == SMALL_BLOCK_SIZE && NN == SMALL_BLOCK_SIZE)
      {
        do_block_small(K, SMALL_BLOCK_SIZE, AA + i * K, SMALL_BLOCK_SIZE, BB, lda, C + i + j * lda);
      }
      else
      {
        // align to small block size and use the function above
        for (int jj = 0; jj < NN; jj++)
        {
          for (int ii = 0; ii < MM; ii++)
          {
            CC[ii + jj * SMALL_BLOCK_SIZE] = C[(ii + i) + (jj + j) * lda];
          }
        }
        do_block_small(K, SMALL_BLOCK_SIZE, AA + i * K, SMALL_BLOCK_SIZE, BB, SMALL_BLOCK_SIZE, CC);

        // write back to C
        for (int jj = 0; jj < NN; jj++)
        {
          for (int ii = 0; ii < MM; ii++)
          {
            C[(ii + i) + (jj + j) * lda] = CC[ii + jj * SMALL_BLOCK_SIZE];
          }
        }
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
