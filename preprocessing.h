#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 256
#define MAX_ROWS 1000  // Adjust this depending on your data size

// Struct to hold the customer ID and rating data
typedef struct {
    int customer_id;
    int rating;
} Rating;

// Function prototype for loading data
int load_data(const char *filename, Rating ratings[], int *count);

#endif // PREPROCESSING_H
