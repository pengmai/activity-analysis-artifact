#pragma once
#include "stdio.h"
#include "stdlib.h"

typedef struct Wishart {
  double gamma;
  int m;
} Wishart;

void dgmm_objective(int d, int k, int n, double const *restrict alphas,
                    double const *restrict dalphas,
                    double const *restrict means, double const *restrict dmeans,
                    double const *restrict icf, double const *restrict dicf,
                    double const *restrict x, double const *restrict dx,
                    Wishart wishart, Wishart *dwishart, double *restrict err,
                    double *restrict derr);

typedef struct _GMMInput {
  int d, k, n;
  double *alphas, *means, *x, *icf;
  double wishart_gamma;
  int wishart_m;
} GMMInput;
