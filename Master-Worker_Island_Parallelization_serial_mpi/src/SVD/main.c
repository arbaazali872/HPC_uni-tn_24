#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svds.h"       // The library providing svds_C_dense

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        // Expect usage: ./svd_mpi <mapped_csv> <num_users> <num_movies>
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <csv_file> <num_users> <num_movies>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank != 0) {
        // "Serial" approach: Only rank 0 does the SVD. Others idle.
        MPI_Finalize();
        return 0;
    }

    // Parse command-line args
    const char* csv_file  = argv[1];
    int num_users  = atoi(argv[2]);
    int num_movies = atoi(argv[3]);

    // 1) Allocate the matrix
    Matrix_C A;
    A.m = num_users;
    A.n = num_movies;
    A.real = (double*) calloc(A.m * A.n, sizeof(double));
    A.imag = (double*) calloc(A.m * A.n, sizeof(double));
    if (!A.real || !A.imag) {
        fprintf(stderr, "Memory allocation failure for matrix.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2) Read the mapped CSV: "user_idx,movie_idx,rating"
    FILE *fp = fopen(csv_file, "r");
    if (!fp) {
        fprintf(stderr, "Could not open file %s.\n", csv_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Reading mapped data from %s...\n", csv_file);

    char line[256];
    // Skip header if necessary (assuming "user_idx,movie_idx,Rating" is a header line)
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp)) {
        int u_idx, m_idx;
        double rating;
        if (sscanf(line, "%d,%d,%lf", &u_idx, &m_idx, &rating) == 3) {
            // Place rating in the dense matrix
            // (We assume 0 <= u_idx < num_users, 0 <= m_idx < num_movies)
            A.real[u_idx * A.n + m_idx] = rating;
        }
    }
    fclose(fp);

    // 3) Call svds_C_dense with default parameters
    double ratio    = 0.0;    // or as needed
    int    kmax     = 5;      // top-5 singular values
    int    v1       = 2;      // solver block size, if relevant
    int    v2       = 2;
    int    n        = (A.m < A.n) ? A.m : A.n; // dimension reference
    FILE  *fout     = stdout; // log output
    int    debug_flg = 0; 
    double tol      = 1e-6;   
    int    maxMv    = 1000;  
    int    info     = 0;

    printf("Starting SVD on matrix %d x %d...\n", A.m, A.n);

    sparse_cplx **result = svds_C_dense(
        &A, ratio, kmax, v1, v2,
        n, fout, debug_flg, tol, maxMv, &info
    );

    if (info != 0) {
        fprintf(stderr, "svds_C_dense returned error code = %d\n", info);
    } else {
        printf("SVD completed successfully. Now do something with 'result'...\n");

        // Typically:
        //   result[0] might hold left singular vectors
        //   result[1] might hold right singular vectors
        //   result[2] might hold singular values
        // Check your `svds.c` to confirm and possibly export them.

        // Then free them properly. E.g.:
        if (result) {
            if (result[0]) free_sparse_cplx(result[0]);
            if (result[1]) free_sparse_cplx(result[1]);
            if (result[2]) free_sparse_cplx(result[2]);
            free(result);
        }
    }

    // 4) Cleanup
    free(A.real);
    free(A.imag);

    MPI_Finalize();
    return 0;
}
