#pragma once
#include <stdio.h>
#include <stdlib.h>

void dlstm_objective(int l, int c, int b, double const *main_params,
                     double *dmain_params, double const *extra_params,
                     double *dextra_params, double *state, double *dstate,
                     double const *sequence, double *dsequence, double *loss,
                     double *dloss);

typedef struct {
  int l, c, b, main_sz, extra_sz, state_sz, seq_sz;
  double *main_params, *extra_params, *state, *sequence;
} LSTMInput;

void read_lstm_instance(const char *data_file, LSTMInput *input);

void free_lstm_instance(LSTMInput *input);
