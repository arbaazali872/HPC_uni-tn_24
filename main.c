#include <stdio.h>
#include "preprocessing.h"
        //   preprocessing

int main() {
    Rating ratings[MAX_ROWS];
    int count = 0;

    if (load_data("combined_data_1.txt", ratings, &count) != 0) {
        return 1;
    }

    printf("First few rows of the data:\n");
    for (int i = 0; i < (count < 5 ? count : 5); i++) {
        printf("Customer ID: %d, Rating: %d\n", ratings[i].customer_id, ratings[i].rating);
    }

    return 0;
}
