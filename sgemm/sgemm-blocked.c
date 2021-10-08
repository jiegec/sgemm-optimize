const char *sgemm_desc = "Simple blocked sgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller sgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, float * restrict A, float * restrict B, float * restrict C)
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
        float cij = C[i + j * lda];
        //cij += A[i + k * lda] * B[k + j * lda];
        cij += A[i + k * lda] * BB[k + j * BLOCK_SIZE];
        C[i + j * lda] = cij;
      }
    }
  }
}

/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_sgemm(int lda, float * restrict A, float * restrict B, float * restrict C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block sgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min(BLOCK_SIZE, lda - i);
        int N = min(BLOCK_SIZE, lda - j);
        int K = min(BLOCK_SIZE, lda - k);

        /* Perform individual block sgemm */
        do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
      }
}
