#pragma once
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Matrix {
  int nrows;
  int ncols;
  double *data;
} Matrix;

typedef struct Triangle {
  int verts[3];
} Triangle;

typedef struct HandModel {
  int n_bones, n_vertices, n_triangles;
  const char **bone_names;
  int *parents;
  double *base_relatives, *inverse_base_absolutes, *base_positions, *weights;
  int *triangles;
  bool is_mirrored;
} HandModel;

typedef struct HandInput {
  HandModel model;
  int *correspondences;
  int n_theta, n_pts;
  double *points, *us, *theta;
  Matrix *brels_mat, *ibabs_mat, *bpos_mat, *weights_mat, *points_mat;
  Triangle *triangles_mat;
} HandInput;

/* Store the converted results to matrices for Enzyme */
struct MatrixConverted {
  Matrix *base_relatives, *inverse_base_absolutes, *base_positions, *weights,
      *points;
  Triangle *triangles;
};

void transpose_in_place(double *matrix, size_t n);

Matrix *ptr_to_matrices(double *data, size_t num_matrices, size_t m, size_t n);

Matrix *ptr_to_matrix(double *data, size_t m, size_t n);

Triangle *ptr_to_triangles(int *data, size_t num_triangles);

void free_matrix_array(Matrix *matrices, size_t num_matrices);

HandModel read_hand_model(const char *model_path, bool transpose);

void free_hand_model(HandModel *model);

HandInput read_hand_data(const char *model_path, const char *data_file,
                         bool complicated, bool transpose);

void free_hand_input(HandInput *input);

void parse_hand_results(const char *ffile, double *J, size_t m, size_t n);

void verify_hand_results(double *ref_J, double *J, size_t m, size_t n,
                         const char *application);

void dhand_objective(double const *theta, double *dtheta, int bone_count,
                     const char **bone_names, const int *parents,
                     Matrix *base_relatives, Matrix *dbase_relatives,
                     Matrix *inverse_base_absolutes,
                     Matrix *dinverse_base_absolutes, Matrix *base_positions,
                     Matrix *dbase_positions, Matrix *weights, Matrix *dweights,
                     const Triangle *triangles, int is_mirrored,
                     int corresp_count, const int *correspondences,
                     Matrix *points, Matrix *dpoints, double *err,
                     double *derr);
