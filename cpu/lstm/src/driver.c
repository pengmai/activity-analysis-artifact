#include "lstm.h"
#include "utils.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>

#define NUM_RUNS 6
#define NUM_REPEATS 4

bool printed = false;
void enzyme_c_lstm_wrapper(LSTMInput *input, double *state) {
  int l = input->l, c = input->c, b = input->b;
  double loss = 0.0, dloss = 1.0;
  double *dmain_params = calloc(input->main_sz, sizeof(double));
  double *dextra_params = calloc(input->extra_sz, sizeof(double));
  double *dstate = calloc(2 * l * b, sizeof(double));
  double *dsequence = calloc(c * b, sizeof(double));
  for (int repeat = 0; repeat < NUM_REPEATS; repeat++) {
    dlstm_objective(l, c, b, input->main_params, dmain_params,
                    input->extra_params, dextra_params, state, dstate,
                    input->sequence, dsequence, &loss, &dloss);
#ifdef VERIFY
    if (!printed) {
      print_d_arr(dmain_params, input->main_sz);
      print_d_arr(dextra_params, input->extra_sz);
      printed = true;
    }
#endif
  }

  free(dmain_params);
  free(dextra_params);
  free(dstate);
  free(dsequence);
}

unsigned long collect_lstm(LSTMInput *input, double *state) {
  struct timeval start, stop;
  memcpy(state, input->state, input->state_sz * sizeof(double));

  gettimeofday(&start, NULL);
  enzyme_c_lstm_wrapper(input, state);
  gettimeofday(&stop, NULL);
  return timediff(start, stop);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <data file>", argv[0]);
    return 1;
  }

  LSTMInput input;
  read_lstm_instance(argv[1], &input);
  double *state = malloc(input.state_sz * sizeof(double));
  unsigned long results_df[NUM_RUNS];
  for (size_t run = 0; run < NUM_RUNS; run++) {
    results_df[run] = collect_lstm(&input, state);
  }
  print_ul_arr(results_df, NUM_RUNS);

  free(state);
  free_lstm_instance(&input);
  return 0;
}
