#pragma once
#ifndef WGSIZE
#define WGSIZE 4
#endif

#define DEFAULT_ITERS 8
#define DEFAULT_NPOSES 65536
#define REF_NPOSES 65536

#define DATA_DIR "./data/bm1"
#define FILE_LIGAND "/ligand.in"
#define FILE_PROTEIN "/protein.in"
#define FILE_FORCEFIELD "/forcefield.in"
#define FILE_POSES "/poses.in"
#define FILE_REF_ENERGIES "/ref_energies.out"

typedef struct {
  float x, y, z;
  int type;
} Atom;

typedef struct {
  int hbtype;
  float radius;
  float hphb;
  float elsc;
} FFParams;

void onecompute(Atom *__restrict__ protein, Atom *__restrict__ ligand,
                float *__restrict__ pose0, float *__restrict__ pose1,
                float *__restrict__ pose2, float *__restrict__ pose3,
                float *__restrict__ pose4, float *__restrict__ pose5,
                float *__restrict__ buffer, FFParams *__restrict forcefield);

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
                  FFParams *__restrict d_forcefield);
