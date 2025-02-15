/*****************************************************************************
 * main.c
 *
 * Reads a CSV "svd_data.csv" (user_id, movie_id, rating),
 * builds a dense matrix A, and calls THU-numbda's partial SVD function:
 *
 *   void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 *
 * Then saves the factor matrices (Uk, Sk, Vk) to a file ("svd_model.dat").
 *
 * To compile with THU-numbda's code (svds.c + deps):
 *   mpicc main.c svds.c LOBPCG_C.c ... -o svd_hpc -lm -lblas -llapack
 *
 * Run:
 *   mpirun -np 1 ./svd_hpc svd_data.csv NUM_ROWS NUM_COLS K
 *****************************************************************************/

 #include <mpi.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 
 /* Suppose THU-numbda's "svds.h" defines:
    - typedef struct mat { int m, n, nnz; doublecomplex *val; int *rowptr, *colind; int format;} mat;
    - doublecomplex (with fields .r, .i)
    - void svds_C_dense(mat *A, mat **Uk, mat **Sk, mat **Vk, int k);
 */
 #include "svds.h"  // Adjust path if needed
 
 /*****************************************************************************
  * fill_dense_matrix_from_csv:
  *   Reads lines "user_id,movie_id,rating" from CSV
  *   and stores them into A->val in row-major order:
  *       A->val[row*A->n + col].r = rating;
  *       A->val[row*A->n + col].i = 0.0;
  *****************************************************************************/
 static int fill_dense_matrix_from_csv(const char *filename, mat *A,
                                       int num_rows, int num_cols)
 {
     if (!A || A->format != 0 || !A->val) {
         fprintf(stderr, "[fill_dense_matrix_from_csv] Invalid input matrix.\n");
         return 1;
     }
 
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         fprintf(stderr, "Error: cannot open CSV file %s\n", filename);
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
             if (uid >= 0 && uid < num_rows && mid >= 0 && mid < num_cols) {
                 int idx = uid * A->n + mid;
                 A->val[idx].r = rating;
                 A->val[idx].i = 0.0;
             }
         }
     }
     fclose(fp);
     return 0;
 }
 
 /*****************************************************************************
  * save_matrices:
  *   Writes Uk, Sk, Vk to a binary file ("svd_model.dat") in this format:
  *     [int Uk_rows] [int Uk_cols]
  *     [Uk_rows*Uk_cols pairs of (double real, double imag)]
  *     [int Sk_rows] [int Sk_cols]
  *     [Sk_rows*Sk_cols pairs (real, imag)]
  *     [int Vk_rows] [int Vk_cols]
  *     [Vk_rows*Vk_cols pairs (real, imag)]
  *
  *   You can later read and reconstruct these matrices as needed.
  *****************************************************************************/
 static void save_matrices(const mat *Uk, const mat *Sk, const mat *Vk)
 {
     FILE *fp = fopen("svd_model.dat", "wb");
     if (!fp) {
         fprintf(stderr, "Warning: Unable to open svd_model.dat for writing.\n");
         return;
     }
 
     // Helper lambda for writing one matrix
     auto write_matrix = [&](const mat *M) {
         if (!M || !M->val) {
             // Indicate 0,0 for dimensions
             int zero = 0;
             fwrite(&zero, sizeof(int), 1, fp);
             fwrite(&zero, sizeof(int), 1, fp);
             return;
         }
         fwrite(&(M->m), sizeof(int), 1, fp);
         fwrite(&(M->n), sizeof(int), 1, fp);
         for (int i = 0; i < M->m * M->n; i++) {
             double r = M->val[i].r;
             double im= M->val[i].i;
             fwrite(&r, sizeof(double), 1, fp);
             fwrite(&im, sizeof(double), 1, fp);
         }
     };
 
     write_matrix(Uk);
     write_matrix(Sk);
     write_matrix(Vk);
 
     fclose(fp);
     printf("[save_matrices] Saved Uk, Sk, Vk to svd_model.dat.\n");
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
         printf("Number of processes: %d (serial HPC approach, only rank 0 does SVD)\n", size);
     }
 
     if (rank == 0) {
         const char *csv_file = argv[1];
         int num_rows = atoi(argv[2]);
         int num_cols = atoi(argv[3]);
         int K        = atoi(argv[4]);
 
         printf("Rank 0: CSV=%s, matrix=%dx%d, partial-SVD rank=%d\n",
                csv_file, num_rows, num_cols, K);
 
         // 1) Allocate a dense mat
         mat A;
         A.m = num_rows;
         A.n = num_cols;
         A.nnz = A.m * A.n;  // for a dense matrix
         A.format = 0;       // 0 => dense
         A.rowptr = NULL;
         A.colind = NULL;
 
         A.val = (doublecomplex*) calloc(A.nnz, sizeof(doublecomplex));
         if (!A.val) {
             fprintf(stderr, "Error: allocation failed for matrix A.\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 2) Fill from CSV
         int err = fill_dense_matrix_from_csv(csv_file, &A, num_rows, num_cols);
         if (err) {
             fprintf(stderr, "Error: fill_dense_matrix_from_csv failed (%d).\n", err);
             free(A.val);
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
 
         // 3) Prepare pointers for SVD output
         mat *Uk = NULL, *Sk = NULL, *Vk = NULL;
 
         // 4) Call THU-numbda’s partial SVD
         svds_C_dense(&A, &Uk, &Sk, &Vk, K);
 
         // 5) Save results to a file
         //    If svds_C_dense allocated Uk->val, Sk->val, Vk->val, 
         //    we can store them in "svd_model.dat".
         save_matrices(Uk, Sk, Vk);
 
         printf("SVD model training completed; results written to svd_model.dat.\n");
 
         // 6) Cleanup
         free(A.val);  // free original matrix
         if (Uk) {
             if (Uk->val) free(Uk->val);
             free(Uk);
         }
         if (Sk) {
             if (Sk->val) free(Sk->val);
             free(Sk);
         }
         if (Vk) {
             if (Vk->val) free(Vk->val);
             free(Vk);
         }
     }
 
     MPI_Finalize();
     return 0;
 }
 