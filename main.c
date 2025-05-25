#include <stdio.h>
#include "preprocessing.h"

int main() {
    size_t num_rows;
    Rating *ratings = load_data("combined_data_1.txt", &num_rows);

    if (!ratings) {
        printf("Error: No data was loaded. Exiting.\n");
        return 1;  
    }

    printf("Successfully loaded %zu rows.\n", num_rows);

    // Print first few rows for verification
    for (size_t i = 0; i < (num_rows < 5 ? num_rows : 5); i++) {
        printf("Customer ID: %d, Rating: %.1f\n", ratings[i].customer_id, ratings[i].rating);
    }

    free(ratings);
    return 0;
}
