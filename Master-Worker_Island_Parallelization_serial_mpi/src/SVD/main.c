/*****************************************************************************
 * main.c
 *
 * This code replicates the idea of reading "svd_data.csv" with columns
 *    user_id,movie_id,rating
 * and performing an SVD — akin to how your Python "Surprise" snippet does it,
 * but here we use THU-numbda's svds_C_dense(...) function:
 *
 *    void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 *
 * from the repo:
 *    https://github.com/THU-numbda/svds-C/blob/main/Codes/AMD/svds.c
 *
 * Compile and link against THU-numbda's code, e.g.:
 *    mpicc main.c svds.c LOBPCG_C.c ... -o svd_hpc -lm -lblas -llapack
 *
 * Usage (serial MPI example):
 *    mpirun -np 1 ./svd_hpc svd_data.csv  NUM_ROWS  NUM_COLS  K
 * Where:
 *  - svd_data.csv has lines: "user_id,movie_id,rating"
 *  - NUM_ROWS, NUM_COLS define the matrix dimension (already pre-mapped IDs)
 *  - K is the rank (# of singular values) to compute
 *****************************************************************************/

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 // Include THU-numbda's headers here. 
 // They must define:
 //   - the 'mat' type (with fields for storing a dense or CSR matrix)
 //   - the function: void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 #include "svds.h"   // or whichever header declares mat and svds_C_dense
 
 /*****************************************************************************
  * In THU-numbda's code, 'mat' often looks like:
  *
  *   typedef struct mat {
  *       int m, n;    // or n, m
  *       int nnz;
  *       doublecomplex *val;  // or double _Complex
  *       int *rowptr, *colind;
  *       int format;  // 0 for dense, 1 for CSR, etc.
  *       ...
  *   } mat;
  *
  * We'll assume 'format=0' means "dense," and we fill 'val' in row-major order.
  *****************************************************************************/
 
 /* A helper to read user_id, movie_id, rating from CSV into a dense matrix 'A'.
    We assume:
      - A->m = num_rows,
      - A->n = num_cols,
      - A->val allocated for A->m * A->n,
      - A->format = 0 for dense,
      - We place rating in A->val[row*A->n + col].r
        and set imaginary part to 0, if using doublecomplex as (r, i).
 */
 static int fill_dense_matrix_from_csv(const char *filename, mat *A,
                                       int num_rows, int num_cols)
 {
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         fprintf(stderr, "Error: cannot open %s\n", filename);
         return 1;
     }
 
     // Optional: if there's a header line "user_id,movie_id,rating", skip it.
     char line[256];
     fgets(line, sizeof(line), fp);
 
     while (fgets(line, sizeof(line), fp)) {
         int uid, mid;
         double rating;
         if (sscanf(line, "%d,%d,%lf", &uid, &mid, &rating) == 3) {
             // assume 0 <= uid < num_rows, 0 <= mid < num_cols
             if (uid < 0 || uid >= num_rows || mid < 0 || mid >= num_cols) {
                 // ignoring out-of-bounds lines
                 continue;
             }
             int idx = uid * A->n + mid; // row-major index
             // 'val' might be an array of 'doublecomplex', e.g. 
             //   val[idx].r = rating;
             //   val[idx].i = 0.0;
             // depending on how 'mat' is actually defined.
             A->val[idx].r = rating;
             A->val[idx].i = 0.0;
         }
     }
     fclose(fp);
     return 0;
 }
 
 int main(int argc, char *argv[])
 {
     MPI_Init(&argc, &argv);
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     if (argc < 5) {
         if (rank == 0) {
             fprintf(stderr, "Usage: %s <svd_data.csv> <NUM_ROWS> <NUM_COLS> <K>\n", argv[0]);
         }
         MPI_Finalize();
         return 1;
     }
 
     if (rank == 0) {
         printf("Running with %d MPI processes (serial approach: only rank 0 does the SVD)\n", size);
     }
 
     // We only do the heavy lifting on rank 0
     if (rank == 0) {
         const char *csv_file = argv[1];
         int num_rows = atoi(argv[2]);
         int num_cols = atoi(argv[3]);
         int K        = atoi(argv[4]);  // # of singular values to compute
 
         printf("Rank 0: reading %s, expecting matrix size %dx%d, K=%d\n",
                csv_file, num_rows, num_cols, K);
 
         // 1) Allocate the mat structure as dense
         mat A;
         // e.g. A->m = num_rows, A->n = num_cols
         A.m = num_rows;
         A.n = num_cols;
         // For dense, format=0
         A.format = 0;
         // nnz = m*n in dense mode
         A.nnz = A.m * A.n;
 
         // rowptr, colind used for CSR, not needed for dense
         A.rowptr = NULL;
         A.colind = NULL;
 
         // Allocate the val array of size nnz
         // In THU-numbda, it might be 'doublecomplex' or a custom struct.
         A.val = (doublecomplex *) calloc(A.nnz, sizeof(doublecomplex));
         if (!A.val) {
             fprintf(stderr, "Allocation error for dense matrix.\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 2) Fill from CSV
         int err = fill_dense_matrix_from_csv(csv_file, &A, num_rows, num_cols);
         if (err) {
             fprintf(stderr, "Failed to load matrix from CSV.\n");
             free(A.val);
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 3) Prepare placeholders for output. 
         //    Typically, Uk, Sk, Vk store left vectors, singular values, right vectors.
         mat *Uk = NULL, *Sk = NULL, *Vk = NULL;
 
         // 4) Call the THU-numbda partial SVD: 
         //    void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
         svds_C_dense(&A, &Uk, &Sk, &Vk, K);
 
         // 5) "Save" the results. 
         //    In Python, you'd do `pickle.dump(svd, file)`. 
         //    Here, you might write Uk->val, Sk->val, Vk->val to a file. E.g.:
         /*
         FILE *fp_model = fopen("svd_model.dat", "wb");
         if (fp_model) {
             // Write dimension info, then data 
             // (the exact format is up to you).
             fclose(fp_model);
         }
         */
 
         printf("SVD model training completed. Now you can save Uk,Sk,Vk.\n");
 
         // 6) Cleanup: free A, Uk, Sk, Vk (depending on how svds_C_dense allocated them).
         free(A.val);
 
         if (Uk) {
             // free(Uk->val);
             // free(Uk);
         }
         if (Sk) {
             // free(Sk->val);
             // free(Sk);
         }
         if (Vk) {
             // free(Vk->val);
             // free(Vk);
         }
     }
 
     MPI_Finalize();
     return 0;
 }
 