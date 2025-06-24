#include "ba.h"
#include "utils.h"
#include <stdio.h>
#include <sys/time.h>

#define NUM_RUNS 6
#define VERIFY_RESULTS 0

double *deadbeef = (double *)0xdeadbeef;

double enzyme_c_compute_w_err(double w) {
  double wb = 0, err = 0, errb = 1.0;
  dcompute_w_error(&w, &wb, &err, &errb);
  return wb;
}

void calculate_reproj_jacobian(BAInput ba_input, BASparseMat *J) {
  double err[2];
  double errb[2];
  double wb = 0.0;
  double *camsb_buf = calloc(BA_NCAMPARAMS, sizeof(double));
  double *Xb_buf = calloc(3, sizeof(double));
  double *dfeats = calloc(2 * ba_input.p, sizeof(double));
  double reproj_err_d[2 * (BA_NCAMPARAMS + 3 + 1)];
  int i;
  for (i = 0; i < ba_input.p; i++) {
    int camIdx = ba_input.obs[2 * i + 0];
    int ptIdx = ba_input.obs[2 * i + 1];

    // Calculate first row
    errb[0] = 1.0;
    errb[1] = 0.0;
    memset(camsb_buf, 0, BA_NCAMPARAMS * sizeof(double));
    memset(Xb_buf, 0, 3 * sizeof(double));
    // memset(dfeats, 0, 2 * ba_input.p);
    wb = 0.0;
    dcompute_reproj_error(&ba_input.cams[camIdx * BA_NCAMPARAMS], camsb_buf,
                          &ba_input.X[ptIdx * 3], Xb_buf, &ba_input.w[i], &wb,
                          &ba_input.feats[i * 2], &dfeats[i * 2], err, errb);

    size_t j = 0;
    for (j = 0; j < BA_NCAMPARAMS; j++) {
      reproj_err_d[j * 2] = camsb_buf[j];
    }
    for (size_t k = 0; k < 3; k++) {
      reproj_err_d[j * 2] = Xb_buf[k];
      j++;
    }
    reproj_err_d[j * 2] = wb;

    errb[1] = 1.0;
    errb[0] = 0.0;
    memset(camsb_buf, 0, BA_NCAMPARAMS * sizeof(double));
    memset(Xb_buf, 0, 3 * sizeof(double));
    // memset(dfeats, 0, 2 * ba_input.p);
    wb = 0.0;
    dcompute_reproj_error(&ba_input.cams[camIdx * BA_NCAMPARAMS], camsb_buf,
                          &ba_input.X[ptIdx * 3], Xb_buf, &ba_input.w[i], &wb,
                          &ba_input.feats[i * 2], &dfeats[i * 2], err, errb);

    j = 0;
    for (j = 0; j < BA_NCAMPARAMS; j++) {
      reproj_err_d[j * 2 + 1] = camsb_buf[j];
    }
    for (size_t k = 0; k < 3; k++) {
      reproj_err_d[j * 2 + 1] = Xb_buf[k];
      j++;
    }
    reproj_err_d[j * 2 + 1] = wb;

    insert_reproj_err_block(J, i, camIdx, ptIdx, reproj_err_d);
  }
  free(camsb_buf);
  free(Xb_buf);
  free(dfeats);
}

void calculate_w_jacobian(BAInput input, BASparseMat *J) {
  for (size_t j = 0; j < input.p; j++) {
    double res = enzyme_c_compute_w_err(input.w[j]);
    insert_w_err_block(J, j, res);
  }
}

unsigned long compute_jacobian(BAInput input, BASparseMat *mat,
                               BASparseMat *ref) {
  struct timeval start, stop;
  clearBASparseMat(mat);
  gettimeofday(&start, NULL);
  calculate_reproj_jacobian(input, mat);
  calculate_w_jacobian(input, mat);
  gettimeofday(&stop, NULL);

  //   print_d_arr(mat->vals, mat->val_end);
  return timediff(start, stop);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <data_file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  BAInput ba_input = read_ba_data(argv[1]);
  int n = ba_input.n, m = ba_input.m, p = ba_input.p;
  BASparseMat mat = initBASparseMat(n, m, p);
  BASparseMat ref = initBASparseMat(n, m, p);

  unsigned long results_df[NUM_RUNS];
  for (size_t run = 0; run < NUM_RUNS; run++) {
    results_df[run] = compute_jacobian(ba_input, &mat, &ref);
  }
  print_ul_arr(results_df, NUM_RUNS);

  free_ba_data(ba_input);
  freeBASparseMat(&mat);
  freeBASparseMat(&ref);
}