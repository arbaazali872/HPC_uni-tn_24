/*
 * serial_svd_recommender.c
 *
 * A serial MPI C program that mimics the Python SVD recommendation workflow.
 *
 * Assumptions:
 *   - The merged data is in "merged_data.txt" (tab-separated, with header).
 *   - The file contains the columns: Cust_Id, Movie_Id, Rating, Genres, Title.
 *   - An SVD function is available from svds-C with the prototype:
 *         int svd(double *A, int m, int n, int k, double *U, double *S, double *V);
 *
 * Compile with:
 *    mpicc -o serial_svd_recommender serial_svd_recommender.c -lm -lsvds
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <mpi.h>
 #include <time.h>
 #include <math.h>
 #include "svds.h"   // Header for the SVD library (assumed to provide svd())
 
 #define MAX_GENRE_LEN 50
 #define MAX_TITLE_LEN 100
 
 // Structure to hold one record of the dataset.
 typedef struct {
     int cust_id;
     int movie_id;
     double rating;
     char genre[MAX_GENRE_LEN];
     char title[MAX_TITLE_LEN];
 } RatingRecord;
 
 // --- Function Prototypes ---
 int load_dataset(const char *filename, RatingRecord **records);
 void build_rating_matrix(RatingRecord *records, int num_records,
                          double **matrix, int num_users, int num_movies);
 void compute_svd(double **matrix, int num_users, int num_movies, int k,
                  double *U_flat, double *S, double *V_flat);
 void reconstruct_ratings(double *U_flat, double *S, double *V_flat,
                          int num_users, int num_movies, int k,
                          double *predicted);
 void free_matrix(double **matrix, int num_rows);
 
 int main(int argc, char *argv[]) {
     MPI_Init(&argc, &argv);
 
     int rank;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
     // This is a serial implementation: only rank 0 will do the work.
     if (rank == 0) {
         double start_time = MPI_Wtime();
 
         /* --- Step 1: Load the Dataset --- */
         RatingRecord *records = NULL;
         int num_records = load_dataset("merged_data.txt", &records);
         if (num_records <= 0) {
             fprintf(stderr, "Error: No records loaded.\n");
             MPI_Abort(MPI_COMM_WORLD, -1);
         }
         printf("Loaded %d records from merged_data.txt\n", num_records);
 
         // For demonstration, we assume the maximum user and movie IDs are known.
         // In practice, you might compute these by scanning the records.
         int num_users = 1000000;  // e.g., 1 million users
         int num_movies = 5000;    // e.g., 5,000 movies
 
         /* --- Step 2: Build the User-Item Rating Matrix --- */
         // Allocate a 2D array (num_users x num_movies)
         double **rating_matrix = malloc(num_users * sizeof(double *));
         for (int i = 0; i < num_users; i++) {
             rating_matrix[i] = calloc(num_movies, sizeof(double));
         }
         build_rating_matrix(records, num_records, rating_matrix, num_users, num_movies);
 
         /* --- Step 3: Compute Truncated SVD --- */
         int k = 50;  // number of latent factors
         // We'll allocate contiguous arrays for U, S, and V.
         double *U_flat = malloc(num_users * k * sizeof(double));
         double *S = malloc(k * sizeof(double));
         double *V_flat = malloc(num_movies * k * sizeof(double));
 
         // Prepare a contiguous copy of the rating matrix as a 1D array in row-major order.
         double *A = malloc(num_users * num_movies * sizeof(double));
         for (int i = 0; i < num_users; i++) {
             for (int j = 0; j < num_movies; j++) {
                 A[i * num_movies + j] = rating_matrix[i][j];
             }
         }
         // Call the SVD routine from the svds-C library.
         if (svd(A, num_users, num_movies, k, U_flat, S, V_flat) != 0) {
             fprintf(stderr, "Error: SVD computation failed.\n");
             MPI_Abort(MPI_COMM_WORLD, -1);
         }
         free(A);
 
         /* --- Step 4: Reconstruct the Predicted Ratings Matrix --- */
         // Allocate a contiguous predicted ratings array (row-major order)
         double *predicted_ratings = malloc(num_users * num_movies * sizeof(double));
         reconstruct_ratings(U_flat, S, V_flat, num_users, num_movies, k, predicted_ratings);
 
         /* --- Step 5: Recommend Movies for a Selected User --- */
         // For example, choose a random user (or a specific user, e.g., 712664)
         srand(time(NULL));
         int user_id = (rand() % num_users) + 1;  // assume 1-indexed user IDs
         printf("\nRecommendations for User %d:\n", user_id);
         // (In a full system, we would filter out movies the user already rated,
         //  group by genre, etc. Here, we simply print the predicted rating for the first 10 movies.)
         for (int j = 0; j < 10; j++) {
             double pred = predicted_ratings[(user_id - 1) * num_movies + j];
             printf("Movie %d predicted rating: %.2f\n", j+1, pred);
         }
 
         double end_time = MPI_Wtime();
         printf("\nTotal computation time: %.2f seconds\n", end_time - start_time);
 
         /* --- Cleanup --- */
         free_matrix(rating_matrix, num_users);
         free(U_flat);
         free(S);
         free(V_flat);
         free(predicted_ratings);
         free(records);
     }
 
     MPI_Finalize();
     return 0;
 }
 
 /* --- Helper Function Implementations --- */
 
 // load_dataset: Reads the file "filename" (assumed tab-separated with header)
 // and returns an array of RatingRecord structures. The number of records is returned.
 int load_dataset(const char *filename, RatingRecord **records) {
     FILE *fp = fopen(filename, "r");
     if (!fp) {
         perror("Error opening dataset file");
         return -1;
     }
 
     char line[512];
     int count = 0;
     // Skip header line.
     if (fgets(line, sizeof(line), fp) == NULL) {
         fclose(fp);
         return -1;
     }
     // First pass: count lines.
     while (fgets(line, sizeof(line), fp) != NULL) {
         count++;
     }
 
     // Allocate records array.
     *records = malloc(count * sizeof(RatingRecord));
     if (!(*records)) {
         fclose(fp);
         return -1;
     }
 
     // Second pass: read data.
     rewind(fp);
     // Skip header.
     fgets(line, sizeof(line), fp);
     int idx = 0;
     while (fgets(line, sizeof(line), fp) != NULL && idx < count) {
         // Expected format: Cust_Id\tMovie_Id\tRating\tGenres\tTitle
         char *token = strtok(line, "\t\n");
         if (token != NULL)
             (*records)[idx].cust_id = atoi(token);
         token = strtok(NULL, "\t\n");
         if (token != NULL)
             (*records)[idx].movie_id = atoi(token);
         token = strtok(NULL, "\t\n");
         if (token != NULL)
             (*records)[idx].rating = atof(token);
         token = strtok(NULL, "\t\n");
         if (token != NULL)
             strncpy((*records)[idx].genre, token, MAX_GENRE_LEN - 1);
         (*records)[idx].genre[MAX_GENRE_LEN - 1] = '\0';
         token = strtok(NULL, "\t\n");
         if (token != NULL)
             strncpy((*records)[idx].title, token, MAX_TITLE_LEN - 1);
         (*records)[idx].title[MAX_TITLE_LEN - 1] = '\0';
 
         idx++;
     }
 
     fclose(fp);
     return count;
 }
 
 // build_rating_matrix: Fills the provided 2D array "matrix" with ratings.
 // Assumes user IDs and movie IDs are 1-indexed.
 void build_rating_matrix(RatingRecord *records, int num_records,
                          double **matrix, int num_users, int num_movies) {
     for (int i = 0; i < num_records; i++) {
         int user = records[i].cust_id - 1;   // 0-indexed
         int movie = records[i].movie_id - 1;
         if (user >= 0 && user < num_users && movie >= 0 && movie < num_movies) {
             matrix[user][movie] = records[i].rating;
         }
     }
 }
 
 // compute_svd: Converts the 2D matrix to a contiguous array and calls the SVD routine.
 // U_flat: (num_users x k), S: (k), V_flat: (num_movies x k)
 // Here we assume that the svd() function from svds-C is available.
 void compute_svd(double **matrix, int num_users, int num_movies, int k,
                  double *U_flat, double *S, double *V_flat) {
     // The matrix A must be contiguous in row-major order.
     double *A = malloc(num_users * num_movies * sizeof(double));
     for (int i = 0; i < num_users; i++) {
         for (int j = 0; j < num_movies; j++) {
             A[i * num_movies + j] = matrix[i][j];
         }
     }
     // Call the SVD routine.
     if (svd(A, num_users, num_movies, k, U_flat, S, V_flat) != 0) {
         fprintf(stderr, "Error in SVD computation.\n");
         free(A);
         exit(-1);
     }
     free(A);
 }
 
 // reconstruct_ratings: Computes predicted = U * diag(S) * V^T.
 // U_flat is of size (num_users x k), V_flat is (num_movies x k), S is length k.
 // The output "predicted" is a contiguous array of size (num_users x num_movies).
 void reconstruct_ratings(double *U_flat, double *S, double *V_flat,
                          int num_users, int num_movies, int k,
                          double *predicted) {
     // For each user i and movie j, compute:
     // predicted[i, j] = sum_{r=0}^{k-1} U_flat[i*k + r] * S[r] * V_flat[j*k + r]
     for (int i = 0; i < num_users; i++) {
         for (int j = 0; j < num_movies; j++) {
             double sum = 0.0;
             for (int r = 0; r < k; r++) {
                 sum += U_flat[i * k + r] * S[r] * V_flat[j * k + r];
             }
             predicted[i * num_movies + j] = sum;
         }
     }
 }
 
 // free_matrix: Frees a 2D array allocated as an array of pointers.
 void free_matrix(double **matrix, int num_rows) {
     for (int i = 0; i < num_rows; i++) {
         free(matrix[i]);
     }
     free(matrix);
 }
 