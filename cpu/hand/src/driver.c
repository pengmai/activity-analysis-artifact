#include "hand.h"
#include "utils.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_RUNS 6
#define VERIFY_RESULTS false

// typedef F64Descriptor1D (*hand_jacobian_row)(HandInput *input, double *derr,
//                                              int row, int col);
bool printed = false;
void hand_jacobian(HandInput *input, double *J) {
  int err_size = 3 * input->n_pts;
  double *dtheta = calloc(input->n_theta, sizeof(double));
  // Extra active stuff
  double *dbase_relatives_ptr =
      calloc(input->model.n_bones * 16, sizeof(double));
  Matrix *dbase_relatives =
      ptr_to_matrices(dbase_relatives_ptr, input->model.n_bones, 4, 4);
  double *dinverse_base_absolutes_ptr =
      calloc(input->model.n_bones * 16, sizeof(double));
  Matrix *dinverse_base_absolutes =
      ptr_to_matrices(dinverse_base_absolutes_ptr, input->model.n_bones, 4, 4);
  double *dbase_positions_ptr =
      calloc(4 * input->model.n_vertices, sizeof(double));
  Matrix *dbase_positions =
      ptr_to_matrix(dbase_positions_ptr, 4, input->model.n_vertices);
  double *dweights_ptr =
      calloc(input->model.n_bones * input->model.n_vertices, sizeof(double));
  Matrix *dweights = ptr_to_matrix(dweights_ptr, input->model.n_bones,
                                   input->model.n_vertices);
  double *dpoints_ptr = calloc(3 * input->n_pts, sizeof(double));
  Matrix *dpoints = ptr_to_matrix(dpoints_ptr, 3, input->n_pts);

  double *err = (double *)malloc(err_size * sizeof(double));
  double *derr = (double *)malloc(err_size * sizeof(double));
  for (size_t i = 0; i < err_size; i++) {
    for (size_t j = 0; j < err_size; j++) {
      derr[j] = (i == j) ? 1.0 : 0.0;
    }
    memset(dtheta, 0, input->n_theta * sizeof(double));
    memset(dbase_relatives_ptr, 0, input->model.n_bones * 16 * sizeof(double));
    memset(dinverse_base_absolutes_ptr, 0,
           input->model.n_bones * 16 * sizeof(double));
    memset(dpoints_ptr, 0, 3 * input->n_pts * sizeof(double));
    memset(dweights_ptr, 0,
           input->model.n_bones * input->model.n_vertices * sizeof(double));
    // Compute a row of the Jacobian
    dhand_objective(input->theta, dtheta, input->model.n_bones,
                    input->model.bone_names, input->model.parents,
                    input->brels_mat, dbase_relatives, input->ibabs_mat,
                    dinverse_base_absolutes, input->bpos_mat, dbase_positions,
                    input->weights_mat, dweights, input->triangles_mat, 0,
                    input->n_pts, input->correspondences, input->points_mat,
                    dpoints, err, derr);

    if (!printed) {
      // print_d_arr(dtheta, input->n_theta);
    }
    // print_d_arr(dpoints_ptr, 3 * input->n_pts);
    // print_d_arr_2d(dinverse_base_absolutes_ptr, input->model.n_bones, 16);
    // print_d_arr_2d(dweights_ptr, input->model.n_bones,
    // input->model.n_vertices);
    for (size_t j = 0; j < input->n_theta; j++) {
      J[i * input->n_theta + j] = dtheta[j];
    }
    // free(dtheta.aligned);
  }
  printed = true;
  free(dtheta);
  free(err);
  free(derr);
}

unsigned long collect_hand(HandInput *input, double *J, double *ref_J) {
  struct timeval start, stop;
  gettimeofday(&start, NULL);
  hand_jacobian(input, J);
  gettimeofday(&stop, NULL);
  //   if (VERIFY_RESULTS) {
  //     verify_hand_results(ref_J, J, 3 * input->n_pts, input->n_theta,
  //     app.name);
  //   }
  return timediff(start, stop);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <model-path> <data-file>\n", argv[0]);
    return 1;
  }
  HandInput input = read_hand_data(argv[1], argv[2], false, true);
  int J_rows = 3 * input.n_pts;
  int J_cols = input.n_theta;
  double *ref_J = (double *)malloc(J_rows * J_cols * sizeof(double));
  //   if (VERIFY_RESULTS) {
  //     populate_ref(&input, ref_J);
  //   }
  double *J = (double *)malloc(J_rows * J_cols * sizeof(double));
  unsigned long results_df[NUM_RUNS] = {0};
  for (size_t run = 0; run < NUM_RUNS; run++) {
    results_df[run] = collect_hand(&input, J, ref_J);
  }
  print_ul_arr(results_df, NUM_RUNS);

  free(ref_J);
  free(J);
}
