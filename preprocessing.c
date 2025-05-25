#include "preprocessing.h"

Rating* load_data(const char *filename, size_t *num_rows) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return NULL;
    }

    // Start with a reasonable capacity
    size_t capacity = 1000000;
    Rating *ratings = malloc(capacity * sizeof(Rating));
    if (!ratings) {
        perror("Memory allocation failed");
        fclose(file);
        return NULL;
    }

    size_t count = 0;
    char line[256];  // Buffer to store each line
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%d,%lf", &ratings[count].customer_id, &ratings[count].rating) == 2) {
            count++;
        }

        // If we reach capacity, stop (prevents realloc issues)
        if (count >= capacity) {
            printf("Warning: Reached max capacity of %zu ratings.\n", capacity);
            break;
        }
    }

    fclose(file);

    if (count == 0) {
        printf("Warning: No valid data was loaded from the file!\n");
        free(ratings);
        return NULL;
    }

    *num_rows = count;
    return ratings;
}
