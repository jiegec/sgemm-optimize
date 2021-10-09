const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, float *restrict A, float *restrict B, float *restrict C)
{
  // pack B to continuous memory in cache
  float BB[BLOCK_SIZE * BLOCK_SIZE];
  for (int j = 0; j < N; ++j)
  {
    for (int k = 0; k < K; ++k)
    {
      BB[k + j * BLOCK_SIZE] = B[k + j * lda];
    }
  }

  for (int k = 0; k < K; ++k)
  {
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
      {
        /* Compute C(i,j) */
        float cij = C[i + j * BLOCK_SIZE];
        //cij += A[i + k * lda] * B[k + j * lda];
        cij += A[i + k * lda] * BB[k + j * BLOCK_SIZE];
        C[i + j * BLOCK_SIZE] = cij;
      }
    }
  }
}

// let compiler optimize for M = N = BLOCK_SIZE
static void do_block_constant_size(int lda, int M, int N, int K, float *restrict A, float *restrict B, float *restrict C)
{
  M = N = BLOCK_SIZE;
  // pack B to continuous memory in cache
  float BB[BLOCK_SIZE * BLOCK_SIZE];
  for (int j = 0; j < N; ++j)
  {
    for (int k = 0; k < K; ++k)
    {
      BB[k + j * BLOCK_SIZE] = B[k + j * lda];
    }
  }

  for (int k = 0; k < K; ++k)
  {
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
      {
        /* Compute C(i,j) */
        float cij = C[i + j * BLOCK_SIZE];
        //cij += A[i + k * lda] * B[k + j * lda];
        cij += A[i + k * lda] * BB[k + j * BLOCK_SIZE];
        C[i + j * BLOCK_SIZE] = cij;
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
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
      int N = min(BLOCK_SIZE, lda - j);

      // pack C to continuous memory in cache
      float CC[BLOCK_SIZE * BLOCK_SIZE];
      for (int jj = 0; jj < N; ++jj)
      {
        for (int ii = 0; ii < M; ++ii)
        {
          CC[ii + jj * BLOCK_SIZE] = C[(ii + i) + (jj + j) * lda];
        }
      }

      /* Accumulate block sgemms into block of C */
      if (M == BLOCK_SIZE && N == BLOCK_SIZE)
      {
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int K = min(BLOCK_SIZE, lda - k);

          /* Perform individual block sgemm */
          do_block_constant_size(lda, M, N, K, A + i + k * lda, B + k + j * lda, CC);
        }
      }
      else
      {
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
          /* Correct block dimensions if block "goes off edge of" the matrix */
          int K = min(BLOCK_SIZE, lda - k);

          /* Perform individual block sgemm */
          do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, CC);
        }
      }

      // unpack
      for (int jj = 0; jj < N; ++jj)
      {
        for (int ii = 0; ii < M; ++ii)
        {
          C[(ii + i) + (jj + j) * lda] = CC[ii + jj * BLOCK_SIZE];
        }
      }
    }
  }
}
