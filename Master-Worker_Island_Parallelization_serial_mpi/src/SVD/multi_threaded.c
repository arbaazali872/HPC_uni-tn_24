/*****************************************************************************
 * main.c
 *
 * 1) Reads a CSV "svd_data.csv" with lines:
 *      user_id,movie_id,rating
 *    mapped so 0 <= user_id < num_rows and 0 <= movie_id < num_cols.
 *
 * 2) Fills a 'mat' structure (from THU-numbda's svds.h), where
 *    mat has fields: int nrows, ncols; double *d;    // stored in row-major order.
 *
 * 3) Calls the function:
 *    void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 *    which is provided by THU-numbda's svds.c (the HPC code, internally using OpenMP).
 *
 * 4) Logs (prints) time taken for each step and a success message.
 *
 * 5) Saves Uk, Sk, Vk to a single binary file "svd_mpi_results.dat" in row-major format.
 *
 * Compilation (example):
 *   gcc -fopenmp main.c svds.c matrix_funs.c -o svd_shared -lm -lopenblas -llapacke
 *
 * Run (shared-memory approach):
 *   ./svd_shared svd_data.csv NUM_ROWS NUM_COLS K
 *****************************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <omp.h>
 #include "svds.h"  // This header includes the definition of mat and the svds_C_dense function
 
 /*****************************************************************************
  * fill_matrix_from_csv:
  *   Reads lines "user_id,movie_id,rating" from 'filename',
  *   and places each rating in row-major order:
  *     A->d[row * A->ncols + col] = rating
  *   Skips one header line if present.
  * Returns 0 on success, nonzero on error.
  *****************************************************************************/
 static int fill_matrix_from_csv(const char *filename, mat *A, int num_rows, int num_cols)
 {
     if (!A || !A->d) {
         fprintf(stderr, "[fill_matrix_from_csv] Error: invalid 'mat' pointer.\n");
         return 1;
     }
     // Check dimensions.
     if (A->nrows != num_rows || A->ncols != num_cols) {
         fprintf(stderr, "[fill_matrix_from_csv] Mismatch: A->nrows/ncols vs. function args.\n");
         return 2;
     }
  
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         fprintf(stderr, "Error: cannot open CSV file '%s'\n", filename);
         return 3;
     }
  
     // Skip header line (if any).
     char line[256];
     if (fgets(line, sizeof(line), fp) == NULL) {
         fclose(fp);
         return 4; // Possibly empty file.
     }
  
     // Read each subsequent line.
     while (fgets(line, sizeof(line), fp)) {
         int uid, mid;
         double rating;
         if (sscanf(line, "%d,%d,%lf", &uid, &mid, &rating) == 3) {
             if (uid >= 0 && uid < num_rows && mid >= 0 && mid < num_cols) {
                 int idx = uid * A->ncols + mid;
                 A->d[idx] = rating;
             }
         }
     }
     fclose(fp);
     return 0;
 }
 
 /*****************************************************************************
  * save_one_matrix:
  *   Helper function to write one matrix M to file fp in binary format.
  *   The function writes two ints (nrows and ncols) followed by M->d.
  *****************************************************************************/
 static void save_one_matrix(const mat *M, FILE *fp) {
     if (!M || !M->d) {
         int zero = 0;
         fwrite(&zero, sizeof(int), 1, fp);
         fwrite(&zero, sizeof(int), 1, fp);
         return;
     }
     fwrite(&M->nrows, sizeof(int), 1, fp);
     fwrite(&M->ncols, sizeof(int), 1, fp);
     fwrite(M->d, sizeof(double), (size_t)(M->nrows * M->ncols), fp);
 }
 
 /*****************************************************************************
  * save_matrices:
  *   Writes Uk, Sk, Vk to "svd_mpi_results.dat" in binary format.
  *****************************************************************************/
 static void save_matrices(const mat *Uk, const mat *Sk, const mat *Vk) {
     FILE *fp = fopen("svd_mpi_results.dat", "wb");
     if (!fp) {
         fprintf(stderr, "[save_matrices] Could not open svd_mpi_results.dat for writing.\n");
         return;
     }
     save_one_matrix(Uk, fp);
     save_one_matrix(Sk, fp);
     save_one_matrix(Vk, fp);
     fclose(fp);
     printf("[save_matrices] Saved Uk, Sk, Vk to 'svd_mpi_results.dat'.\n");
 }
 
 /*****************************************************************************
  * main:
  *   Pure shared-memory version using OpenMP + multi-threaded BLAS.
  *****************************************************************************/
 int main(int argc, char *argv[]) {
     double total_time_start = omp_get_wtime();
 
     if (argc < 5) {
         fprintf(stderr, "Usage: %s <csv_file> <num_rows> <num_cols> <K>\n", argv[0]);
         return 1;
     }
 
     const char *csv_file = argv[1];
     int num_rows = atoi(argv[2]);
     int num_cols = atoi(argv[3]);
     int K = atoi(argv[4]);
 
     printf("Building matrix from '%s' with dimensions %dx%d, computing rank-%d truncated SVD.\n", csv_file, num_rows, num_cols, K);
 
     // 1) Allocate the dense matrix A.
     mat A;
     A.nrows = num_rows;
     A.ncols = num_cols;
     A.d = (double*) calloc((size_t)(A.nrows * A.ncols), sizeof(double));
     if (!A.d) {
         fprintf(stderr, "Error: allocation failed for A->d\n");
         return 1;
     }
 
     // 2) Fill matrix from CSV.
     double t_csv = omp_get_wtime();
     int err = fill_matrix_from_csv(csv_file, &A, num_rows, num_cols);
     t_csv = omp_get_wtime() - t_csv;
     if (err) {
         fprintf(stderr, "Error reading CSV (code=%d)\n", err);
         free(A.d);
         return 1;
     }
     printf("CSV reading & matrix filling took %.6f sec.\n", t_csv);
 
     // 3) Prepare placeholders for the SVD outputs.
     mat *Uk = NULL, *Sk = NULL, *Vk = NULL;
 
     // 4) Compute the truncated SVD.
     double t_svd = omp_get_wtime();
     svds_C_dense(&A, &Uk, &Sk, &Vk, K); // This function is internally parallelized.
     t_svd = omp_get_wtime() - t_svd;
     printf("SVD computation took %.6f sec.\n", t_svd);
 
     // 5) Save the SVD results to a binary file.
     double t_save = omp_get_wtime();
     save_matrices(Uk, Sk, Vk);
     t_save = omp_get_wtime() - t_save;
     printf("Saving Uk, Sk, Vk took %.6f sec.\n", t_save);
 
     // 6) Free memory.
     free(A.d);
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
 
     double total_time = omp_get_wtime() - total_time_start;
     printf("Total program time: %.6f sec.\n", total_time);
 
     return 0;
 }
 