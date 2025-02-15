/*****************************************************************************
 * main.c
 *
 * Reads a CSV "svd_data.csv" containing lines:
 *     user_id,movie_id,rating
 * already mapped so that 0 <= user_id < num_rows and 0 <= movie_id < num_cols.
 *
 * Builds a dense matrix A (row-major) in a struct "mat" with fields (m, n, data).
 *
 * Then calls the function:
 *   void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 * as defined in THU-numbda's code (or your local "svds.c").
 *
 * Finally, saves Uk, Sk, Vk to a file "svd_model.dat" with no placeholders.
 *
 * Compile (example):
 *   mpicc main.c svds.c -o svd_mpi -lm -lblas -llapack
 * or include other THU-numbda files (LOBPCG_C.c, etc.) as needed.
 *
 * Run (serial MPI):
 *   mpirun -np 1 ./svd_mpi svd_data.csv NUM_ROWS NUM_COLS K
 *****************************************************************************/

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 /* We assume your 'svds.h' declares something like:
      typedef struct {
        int m;
        int n;
        double *data;  // row-major
      } mat;
 
    and the SVD function:
      void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 */
 #include "svds.h"
 
 
 /*****************************************************************************
  * fill_matrix_from_csv:
  *   Reads lines "user_id,movie_id,rating" from 'filename',
  *   and places each rating into A->data[row*A->n + col].
  *
  *   We skip a possible header line. 
  *
  * Return 0 on success, nonzero on error.
  *****************************************************************************/
 static int fill_matrix_from_csv(const char *filename, mat *A,
                                 int num_rows, int num_cols)
 {
     if (!A || !A->data) {
         fprintf(stderr, "[fill_matrix_from_csv] Invalid mat.\n");
         return 1;
     }
 
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         fprintf(stderr, "Error: cannot open %s\n", filename);
         return 2;
     }
 
     // If there's a header line, skip it
     char line[256];
     if (fgets(line, sizeof(line), fp) == NULL) {
         // Possibly empty file
         fclose(fp);
         return 3;
     }
 
     while (fgets(line, sizeof(line), fp)) {
         int uid, mid;
         double rating;
         if (sscanf(line, "%d,%d,%lf", &uid, &mid, &rating) == 3) {
             // Check bounds
             if (uid >= 0 && uid < num_rows && mid >= 0 && mid < num_cols) {
                 int idx = uid * A->n + mid;
                 A->data[idx] = rating;
             }
         }
     }
     fclose(fp);
     return 0;
 }
 
 /*****************************************************************************
  * save_matrices:
  *   Writes Uk, Sk, Vk to a binary file "svd_model.dat" with the format:
  *
  *   [Uk->m] [Uk->n]
  *   (Uk->m * Uk->n) doubles
  *   [Sk->m] [Sk->n]
  *   (Sk->m * Sk->n) doubles
  *   [Vk->m] [Vk->n]
  *   (Vk->m * Vk->n) doubles
  *
  * Return: none; prints a message on success.
  *****************************************************************************/
 static void save_matrices(const mat *Uk, const mat *Sk, const mat *Vk)
 {
     FILE *fp = fopen("svd_model.dat", "wb");
     if (!fp) {
         fprintf(stderr, "[save_matrices] Could not open svd_model.dat for writing.\n");
         return;
     }
 
     // Helper to write a single matrix
     // If M is NULL, we store zero for dimensions
     // otherwise, we store M->m, M->n, then M->data in row-major order
     // (M->m * M->n) doubles
     int m_val = 0, n_val = 0;
     const double *pdata = NULL;
 
     // Writes out the matrix in a <rows> <cols> <data...> binary format
     auto write_mat = [&](const mat *M) {
         if (!M || !M->data) {
             int zero = 0;
             fwrite(&zero, sizeof(int), 1, fp);
             fwrite(&zero, sizeof(int), 1, fp);
             return;
         }
         fwrite(&M->m, sizeof(int), 1, fp);
         fwrite(&M->n, sizeof(int), 1, fp);
         fwrite(M->data, sizeof(double), M->m * M->n, fp);
     };
 
     write_mat(Uk);
     write_mat(Sk);
     write_mat(Vk);
 
     fclose(fp);
     printf("[save_matrices] Matrices saved to svd_model.dat\n");
 }
 
 int main(int argc, char *argv[])
 {
     MPI_Init(&argc, &argv);
 
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     if (argc < 5) {
         if (rank == 0) {
             fprintf(stderr, "Usage: %s <csv> <num_rows> <num_cols> <K>\n", argv[0]);
         }
         MPI_Finalize();
         return 1;
     }
 
     if (rank == 0) {
         printf("MPI size = %d (serial: only rank 0 does the SVD)\n", size);
     }
 
     if (rank == 0) {
         const char *csv_file = argv[1];
         int num_rows = atoi(argv[2]);
         int num_cols = atoi(argv[3]);
         int K        = atoi(argv[4]);
 
         printf("Rank 0: reading %s, building %dx%d matrix, requesting top-%d SVD.\n",
                csv_file, num_rows, num_cols, K);
 
         // 1) Allocate the mat
         mat A;
         A.m = num_rows;
         A.n = num_cols;
         A.data = (double*) calloc((size_t)A.m * A.n, sizeof(double));
         if (!A.data) {
             fprintf(stderr, "Allocation failed for matrix A.\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 2) Fill from CSV
         int err = fill_matrix_from_csv(csv_file, &A, A.m, A.n);
         if (err) {
             fprintf(stderr, "Error reading CSV (code=%d)\n", err);
             free(A.data);
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 3) Prepare placeholders for output
         mat *Uk = NULL, *Sk = NULL, *Vk = NULL;
 
         // 4) Call partial SVD (implemented in svds.c or other THU-numbda files)
         svds_C_dense(&A, &Uk, &Sk, &Vk, K);
 
         // 5) Save the results (Uk, Sk, Vk) => "svd_model.dat"
         save_matrices(Uk, Sk, Vk);
 
         printf("SVD completed. The factor matrices have been written to svd_model.dat\n");
 
         // 6) Cleanup
         free(A.data);
         if (Uk) {
             if (Uk->data) free(Uk->data);
             free(Uk);
         }
         if (Sk) {
             if (Sk->data) free(Sk->data);
             free(Sk);
         }
         if (Vk) {
             if (Vk->data) free(Vk->data);
             free(Vk);
         }
     }
 
     MPI_Finalize();
     return 0;
 }
 