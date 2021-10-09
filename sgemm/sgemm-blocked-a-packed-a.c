const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#if !defined(SMALL_BLOCK_SIZE)
#define SMALL_BLOCK_SIZE 4
#endif

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
  // local buffer to numbers in C
  float CC[SMALL_BLOCK_SIZE * SMALL_BLOCK_SIZE];

  // pack
  for (int j = 0; j < SMALL_BLOCK_SIZE; ++j)
  {
    for (int i = 0; i < SMALL_BLOCK_SIZE; ++i)
    {
      CC[i + j * SMALL_BLOCK_SIZE] = C[i + j * ldc];
    }
  }

  for (int k = 0; k < K; ++k)
  {
    /* For each column j of B */
    for (int j = 0; j < SMALL_BLOCK_SIZE; ++j)
    {
      /* For each row i of A */
      for (int i = 0; i < SMALL_BLOCK_SIZE; ++i)
      {
        /* Compute C(i,j) */
        float cij = CC[i + j * SMALL_BLOCK_SIZE];
        cij += A[i + k * lda] * B[k + j * ldb];
        CC[i + j * SMALL_BLOCK_SIZE] = cij;
      }
    }
  }

  // unpack
  for (int j = 0; j < SMALL_BLOCK_SIZE; ++j)
  {
    for (int i = 0; i < SMALL_BLOCK_SIZE; ++i)
    {
      C[i + j * ldc] = CC[i + j * SMALL_BLOCK_SIZE];
    }
  }
}

// two level blocking
// A: MxK, B: KxN, C: MxN
static void do_block_large(int M, int N, int K, int lda, float *restrict A, int ldb, float *restrict B, int ldc, float *restrict C)
{
  float AA[BLOCK_SIZE * SMALL_BLOCK_SIZE];

  /* For each block-row of C */
  for (int i = 0; i < M; i += SMALL_BLOCK_SIZE)
  {
    int MM = min(SMALL_BLOCK_SIZE, M - i);
    /* For each block-column of C */
    for (int j = 0; j < N; j += SMALL_BLOCK_SIZE)
    {
      int NN = min(SMALL_BLOCK_SIZE, N - j);

      if (MM == SMALL_BLOCK_SIZE && NN == SMALL_BLOCK_SIZE)
      {
        /* Perform individual block sgemm */
        // pack A
        for (int jj = 0; jj < K; jj++)
        {
          for (int ii = 0; ii < SMALL_BLOCK_SIZE; ii++)
          {
            AA[ii + jj * SMALL_BLOCK_SIZE] = A[(ii + i) + jj * lda];
          }
        }

        do_block_small(K, SMALL_BLOCK_SIZE, AA, lda, B + 0 + j * lda, lda, C + i + j * lda);
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
