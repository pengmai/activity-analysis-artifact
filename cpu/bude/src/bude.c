#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "bude.h"
#include "vec-pose-inner.c"

extern struct {
  int natlig;
  int natpro;
  int ntypes;
  int nposes;
  Atom *restrict protein;
  Atom *restrict ligand;
  FFParams *restrict forcefield;
  float *restrict poses[6];
  char *deckDir;
  int iterations;
} params;

void __enzyme_autodiff(void *, ...);
int enzyme_const;
int enzyme_dup;

// Only protein and results are active
void fasten_main(const int natlig, const int natpro,
                 const Atom *restrict protein, const Atom *restrict ligand,
                 const float *restrict transforms_0,
                 const float *restrict transforms_1,
                 const float *restrict transforms_2,
                 const float *restrict transforms_3,
                 const float *restrict transforms_4,
                 const float *restrict transforms_5, float *restrict results,
                 const FFParams *restrict forcefield, const int group);

void compute(Atom *__restrict__ protein, Atom *__restrict__ ligand,
             float *__restrict__ poses[6], float *__restrict__ buffer,
             FFParams *__restrict forcefield) {
  int iters = params.iterations;
  int npose = params.nposes;
  for (int itr = 0; itr < iters; itr++) {
    for (unsigned group = 0; group < (npose / WGSIZE); group++) {
      fasten_main(params.natlig, params.natpro, protein, ligand, poses[0],
                  poses[1], poses[2], poses[3], poses[4], poses[5], buffer,
                  forcefield, group);
    }
  }
}

void onecompute(Atom *__restrict__ protein, Atom *__restrict__ ligand,
                float *__restrict__ pose0, float *__restrict__ pose1,
                float *__restrict__ pose2, float *__restrict__ pose3,
                float *__restrict__ pose4, float *__restrict__ pose5,
                float *__restrict__ buffer, FFParams *__restrict forcefield) {
  int npose = params.nposes;
  for (unsigned group = 0; group < (npose / WGSIZE); group++) {
    fasten_main(params.natlig, params.natpro, protein, ligand, pose0, pose1,
                pose2, pose3, pose4, pose5, buffer, forcefield, group);
  }
}

void done_compute(Atom *__restrict__ protein, Atom *__restrict d_protein,
                  Atom *__restrict__ ligand, Atom *__restrict__ d_ligand,
                  float *__restrict__ pose0, float *__restrict__ d_pose0,
                  float *__restrict__ pose1, float *__restrict__ d_pose1,
                  float *__restrict__ pose2, float *__restrict__ d_pose2,
                  float *__restrict__ pose3, float *__restrict__ d_pose3,
                  float *__restrict__ pose4, float *__restrict__ d_pose4,
                  float *__restrict__ pose5, float *__restrict__ d_pose5,
                  float *__restrict__ buffer, float *__restrict__ d_buffer,
                  FFParams *__restrict forcefield,
                  FFParams *__restrict d_forcefield) {
#if ALL_ACTIVE
  // clang-format off
  __enzyme_autodiff(
     (void *)onecompute,
      enzyme_dup, protein, d_protein,
      enzyme_dup, ligand, d_ligand,
      enzyme_dup, pose0, d_pose0,
      enzyme_dup, pose1, d_pose1,
      enzyme_dup, pose2, d_pose2,
      enzyme_dup, pose3, d_pose3,
      enzyme_dup, pose4, d_pose4,
      enzyme_dup, pose5, d_pose5,
      enzyme_dup, buffer, d_buffer,
      // In this benchmark, marking forcefield as active is the input that has
      // the biggest impact
      enzyme_dup, forcefield, d_forcefield);
// clang-format on
#else
  // default activity
  __enzyme_autodiff((void *)onecompute, enzyme_dup, protein, d_protein,
                    enzyme_const, ligand, enzyme_const, pose0, enzyme_const,
                    pose1, enzyme_const, pose2, enzyme_const, pose3,
                    enzyme_const, pose4, enzyme_const, pose5, enzyme_dup,
                    buffer, d_buffer, enzyme_const, forcefield);
#endif
}
