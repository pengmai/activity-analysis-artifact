#include "lstm.h"

void read_lstm_instance(const char *data_file, LSTMInput *input) {
  FILE *fd = fopen(data_file, "r");
  if (!fd) {
    fprintf(stderr, "Failed to open file: %s\n", data_file);
    exit(1);
  }
  fscanf(fd, "%i %i %i", &input->l, &input->c, &input->b);
  int l = input->l, c = input->c, b = input->b;
  int main_sz = 2 * l * 4 * b;
  int extra_sz = 3 * b;
  int state_sz = 2 * l * b;
  int seq_sz = c * b;
  input->main_sz = main_sz;
  input->extra_sz = extra_sz;
  input->state_sz = state_sz;
  input->seq_sz = seq_sz;

  input->main_params = (double *)malloc(main_sz * sizeof(double));
  input->extra_params = (double *)malloc(extra_sz * sizeof(double));
  input->state = (double *)malloc(state_sz * sizeof(double));
  input->sequence = (double *)malloc(seq_sz * sizeof(double));

  for (int i = 0; i < main_sz; i++) {
    fscanf(fd, "%lf", &input->main_params[i]);
  }

  for (int i = 0; i < extra_sz; i++) {
    fscanf(fd, "%lf", &input->extra_params[i]);
  }

  for (int i = 0; i < state_sz; i++) {
    fscanf(fd, "%lf", &input->state[i]);
  }

  for (int i = 0; i < seq_sz; i++) {
    fscanf(fd, "%lf", &input->sequence[i]);
  }

  fclose(fd);
}

void free_lstm_instance(LSTMInput *input) {
  free(input->main_params);
  free(input->extra_params);
  free(input->state);
  free(input->sequence);
}