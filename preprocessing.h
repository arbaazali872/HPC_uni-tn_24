#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdio.h>
#include <stdlib.h>

// Struct for storing customer ratings
typedef struct {
    int customer_id;
    double rating;  // Changed to double for float64 equivalent
} Rating;

// Function prototype to load data dynamically
Rating* load_data(const char *filename, size_t *num_rows);

#endif // PREPROCESSING_H
