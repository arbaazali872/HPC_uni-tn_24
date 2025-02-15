/*****************************************************************************
 * main.c
 *
 * 1) Reads a CSV "svd_data.csv" with lines:
 *      user_id,movie_id,rating
 *    mapped so 0 <= user_id < num_rows and 0 <= movie_id < num_cols.
 *
 * 2) Fills a 'mat' structure (from THU-numbda's svds.h), where
 *    mat has fields: int nrows, ncols; double *d;
 *
 * 3) Calls the function:
 *       void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 *    which is provided by THU-numbda's svds.c (HPC code).
 *
 * 4) Prints a success message. (No matrix saving or placeholders.)
 *
 * Compilation (example):
 *   mpicc main.c svds.c -o svd_mpi -lm -lblas -llapack
 *
 * Run (serial approach):
 *   mpirun -np 1 ./svd_mpi svd_data.csv NUM_ROWS NUM_COLS K
 *****************************************************************************/

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include "svds.h" 
 
 /* 
    Include THU-numbda's "svds.h", which you said contains:
      typedef struct {
          int nrows, ncols;
          double *d;
      } mat;
 
    and the function prototype:
      void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 */
 #include "svds.h"
 
 
 /*****************************************************************************
  * fill_matrix_from_csv:
  *   Reads lines "user_id,movie_id,rating" from 'filename',
  *   and places each rating in row-major order:
  *     A->d[row * A->ncols + col] = rating
  *   Skips one header line if present.
  *
  * Return 0 on success, nonzero on error.
  *****************************************************************************/
 static int fill_matrix_from_csv(const char *filename, mat *A,
                                 int num_rows, int num_cols)
 {
     if (!A || !A->d) {
         fprintf(stderr, "[fill_matrix_from_csv] Error: invalid 'mat' pointer.\n");
         return 1;
     }
     // Double-check that A->nrows, A->ncols match num_rows, num_cols
     if (A->nrows != num_rows || A->ncols != num_cols) {
         fprintf(stderr, "[fill_matrix_from_csv] Mismatch: A->nrows/ncols vs. function args.\n");
         return 2;
     }
 
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         fprintf(stderr, "Error: cannot open CSV file %s\n", filename);
         return 3;
     }
 
     // If there's a header line, skip it
     char line[256];
     if (fgets(line, sizeof(line), fp) == NULL) {
         fclose(fp);
         return 4; // Possibly empty file
     }
 
     // Read each subsequent line
     while (fgets(line, sizeof(line), fp)) {
         int uid, mid;
         double rating;
         if (sscanf(line, "%d,%d,%lf", &uid, &mid, &rating) == 3) {
             if (uid >= 0 && uid < num_rows &&
                 mid >= 0 && mid < num_cols) {
                 int idx = uid * A->ncols + mid;
                 A->d[idx] = rating;
             }
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
             fprintf(stderr, "Usage: %s <csv_file> <num_rows> <num_cols> <K>\n", argv[0]);
         }
         MPI_Finalize();
         return 1;
     }
 
     if (rank == 0) {
         printf("MPI size=%d (serial HPC approach: only rank 0 does the SVD)\n", size);
     }
 
     // We'll only do the heavy lifting on rank 0
     if (rank == 0) {
         const char *csv_file = argv[1];
         int num_rows = atoi(argv[2]);
         int num_cols = atoi(argv[3]);
         int K        = atoi(argv[4]);
 
         printf("Rank 0: reading %s, creating matrix %dx%d, K=%d\n",
                csv_file, num_rows, num_cols, K);
 
         // 1) Allocate the mat
         mat A;
         A.nrows = num_rows;
         A.ncols = num_cols;
         A.d = (double*) calloc((size_t)(A.nrows * A.ncols), sizeof(double));
         if (!A.d) {
             fprintf(stderr, "Error: allocation failed for A.d\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 2) Fill from CSV
         int err = fill_matrix_from_csv(csv_file, &A, A.nrows, A.ncols);
         if (err) {
             fprintf(stderr, "Error reading CSV (code=%d)\n", err);
             free(A.d);
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 3) Prepare placeholders for the HPC partial SVD outputs
         mat *Uk = NULL, *Sk = NULL, *Vk = NULL;
 
         // 4) Call THU-numbda's HPC partial SVD function
         //    Implementation is in svds.c, not here
         svds_C_dense(&A, &Uk, &Sk, &Vk, K);
 
         // 5) Print a success message
         printf("svds_C_dense call completed successfully.\n");
         // For now, we do NOT save or do anything with Uk,Sk,Vk. We just confirm it's done.
 
         // 6) Cleanup
         free(A.d);
 
         // If you want to avoid memory leaks, free HPC allocations if needed
         // (assuming the HPC code allocates Uk->d, etc.)
         if (Uk) {
             if (Uk->d) free(Uk->d);
             free(Uk);
         }
         if (Sk) {
             if (Sk->d) free(Sk->d);
             free(Sk);
         }
         if (Vk) {
             if (Vk->d) free(Vk->d);
             free(Vk);
         }
     }
 
     MPI_Finalize();
     return 0;
 }
 