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
 *    void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 *    which is provided by THU-numbda's svds.c (HPC code).
 *
 * 4) Logs time taken for each step and a success message to "svd_mpi.log".
 *
 * 5) Saves Uk, Sk, Vk to a single binary file "svd_mpi_results.dat" in row-major format.
 *
 * Compilation (example):
 *   mpicc main.c svds.c matrix_funcs.c -o svd_mpi -lm -lblas -llapack
 * or include other THU-numbda files (LOBPCG_C.c, etc.) as needed.
 *
 * Run (serial HPC approach):
 *   mpirun -np 1 ./svd_mpi svd_data.csv NUM_ROWS NUM_COLS K
 *****************************************************************************/

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 /*
    Include THU-numbda's "svds.h", which itself includes "matrix_funcs.h".
    'mat' is defined as:
      typedef struct {
          int nrows, ncols;
          double * d;    // row-major
      } mat;
 
    and:
      void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 */
 #include "svds.h"
 
 /*****************************************************************************
  * fill_matrix_from_csv:
  *   Reads lines "user_id,movie_id,rating" from 'filename',
  *   places each rating in row-major order:
  *     A->d[row * A->ncols + col] = rating
  *   Skips one header line if present.
  * Return 0 on success, nonzero on error.
  *****************************************************************************/
 static int fill_matrix_from_csv(const char *filename, mat *A,
                                 int num_rows, int num_cols)
 {
     if (!A || !A->d) {
         fprintf(stderr, "[fill_matrix_from_csv] Error: invalid 'mat' pointer.\n");
         return 1;
     }
     // Double-check dimensions
     if (A->nrows != num_rows || A->ncols != num_cols) {
         fprintf(stderr, "[fill_matrix_from_csv] Mismatch: A->nrows/ncols vs. function args.\n");
         return 2;
     }
 
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         fprintf(stderr, "Error: cannot open CSV file '%s'\n", filename);
         return 3;
     }
 
     // Skip the first line if it's a header
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
                 mid >= 0 && mid < num_cols)
             {
                 int idx = uid * A->ncols + mid;
                 A->d[idx] = rating;
             }
         }
     }
     fclose(fp);
     return 0;
 }
 
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

 static void save_matrices(const mat *Uk, const mat *Sk, const mat *Vk)
{
    FILE *fp = fopen("svd_mpi_results.dat", "wb");
    if (!fp) {
        fprintf(stderr, "[save_matrices] Could not open svd_mpi_results.dat for writing.\n");
        return;
    }

    save_one_matrix(Uk, fp);
    save_one_matrix(Sk, fp);
    save_one_matrix(Vk, fp);

    fclose(fp);
    printf("[save_matrices] Wrote Uk, Sk, Vk to 'svd_mpi_results.dat'.\n");
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
 
     if (rank == 0) {
         // Timing
         double total_time_start = MPI_Wtime();
 
         // Parse arguments
         const char *csv_file = argv[1];
         int num_rows = atoi(argv[2]);
         int num_cols = atoi(argv[3]);
         int K        = atoi(argv[4]);
 
         // Open log file
         FILE *log_fp = fopen("svd_mpi.log", "w");
         if (!log_fp) {
             fprintf(stderr, "Error: cannot open svd_mpi.log for writing.\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         fprintf(log_fp, "Rank 0: reading '%s', building %dx%d matrix, K=%d\n",
                 csv_file, num_rows, num_cols, K);
 
         // 1) Allocate mat A
         mat A;
         A.nrows = num_rows;
         A.ncols = num_cols;
         A.d = (double*) calloc((size_t)(A.nrows * A.ncols), sizeof(double));
         if (!A.d) {
             fprintf(log_fp, "Error: allocation failed for A->d\n");
             fclose(log_fp);
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 2) Fill matrix from CSV
         double t_csv = MPI_Wtime();
         int err = fill_matrix_from_csv(csv_file, &A, A.nrows, A.ncols);
         t_csv = MPI_Wtime() - t_csv;
         if (err) {
             fprintf(log_fp, "Error reading CSV (code=%d)\n", err);
             free(A.d);
             fclose(log_fp);
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
         fprintf(log_fp, "CSV reading & matrix fill took %.6f sec.\n", t_csv);
 
         // 3) Prepare placeholders for HPC partial SVD
         mat *Uk = NULL, *Sk = NULL, *Vk = NULL;
 
         // 4) SVD
         double t_svd = MPI_Wtime();
         svds_C_dense(&A, &Uk, &Sk, &Vk, K); // HPC code from svds.c
         t_svd = MPI_Wtime() - t_svd;
         fprintf(log_fp, "SVD computation took %.6f sec.\n", t_svd);
 
         // 5) Save the SVD results to a file
         double t_save = MPI_Wtime();
         save_matrices(Uk, Sk, Vk);
         t_save = MPI_Wtime() - t_save;
         fprintf(log_fp, "Saving Uk, Sk, Vk took %.6f sec.\n", t_save);
 
         // 6) Free memory
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
 
         double total_time = MPI_Wtime() - total_time_start;
         fprintf(log_fp, "Total program time: %.6f sec.\n", total_time);
 
         // Done
         fprintf(log_fp, "svds_C_dense call completed successfully!\n");
         fclose(log_fp);
     }
 
     MPI_Finalize();
     return 0;
 }
 