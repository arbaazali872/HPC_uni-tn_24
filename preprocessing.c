#include "preprocessing.h"

int load_data(const char *filename, Rating ratings[], int *count) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    *count = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file) && *count < MAX_ROWS) {
        if (sscanf(line, "%d,%d", &ratings[*count].customer_id, &ratings[*count].rating) == 2) {
            (*count)++;
        }
    }

    fclose(file);
    return 0;
}
