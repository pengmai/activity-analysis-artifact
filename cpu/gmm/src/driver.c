#include "gmm.h"

#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

GMMInput read_gmm_data(const char *data_file) {
  FILE *fp;
  GMMInput gmm_input;
  fp = fopen(data_file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file \"%s\"\n", data_file);
    exit(EXIT_FAILURE);
  }

  int d, k, n;
  fscanf(fp, "%d %d %d", &d, &k, &n);

  int icf_sz = d * (d + 1) / 2;
  double *alphas = (double *)malloc(k * sizeof(double));
  double *means = (double *)malloc(d * k * sizeof(double));
  double *icf = (double *)malloc(icf_sz * k * sizeof(double));
  double *x = (double *)malloc(d * n * sizeof(double));

  for (int i = 0; i < k; i++) {
    fscanf(fp, "%lf", &alphas[i]);
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < d; j++) {
      fscanf(fp, "%lf", &means[i * d + j]);
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < icf_sz; j++) {
      fscanf(fp, "%lf", &icf[i * icf_sz + j]);
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      fscanf(fp, "%lf", &x[i * d + j]);
    }
  }

  int wishart_m;
  double wishart_gamma;
  fscanf(fp, "%lf %d", &wishart_gamma, &wishart_m);
  fclose(fp);

  gmm_input.d = d;
  gmm_input.k = k;
  gmm_input.n = n;
  gmm_input.alphas = alphas;
  gmm_input.means = means;
  gmm_input.icf = icf;
  gmm_input.x = x;
  gmm_input.wishart_gamma = wishart_gamma;
  gmm_input.wishart_m = wishart_m;
  return gmm_input;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: %s <data file>\n", argv[0]);
    return 1;
  }
  GMMInput gmm_input = read_gmm_data(argv[1]);
  int d = gmm_input.d, k = gmm_input.k, n = gmm_input.n;
  int icf_sz = d * (d + 1) / 2;

  double err = 0.0, derr = 1.0;
  Wishart wishart = {.gamma = gmm_input.wishart_gamma,
                     .m = gmm_input.wishart_m};
  Wishart dwishart = {.gamma = 0.0, .m = wishart.m};
  double *dalphas = calloc(k, sizeof(double));
  double *dmeans = calloc(d * k, sizeof(double));
  double *dicf = calloc(icf_sz * k, sizeof(double));
  double *dx = calloc(d * n, sizeof(double));

  struct timeval start, stop;
  const int NUM_RUNS = 6;
  unsigned long results[NUM_RUNS];
  for (unsigned run = 0; run < NUM_RUNS; run++) {
    memset(dalphas, 0, k * sizeof(double));
    memset(dmeans, 0, d * k * sizeof(double));
    memset(dicf, 0, icf_sz * k * sizeof(double));
    memset(dx, 0, d * n * sizeof(double));
    dwishart.gamma = 0.0;
    derr = 1.0;

    gettimeofday(&start, NULL);
    dgmm_objective(d, k, n, gmm_input.alphas, dalphas, gmm_input.means, dmeans,
                   gmm_input.icf, dicf, gmm_input.x, dx, wishart, &dwishart,
                   &err, &derr);
    gettimeofday(&stop, NULL);
    results[run] = timediff(start, stop);
  }

  print_ul_arr(results, NUM_RUNS);
}
