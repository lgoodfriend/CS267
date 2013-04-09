//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
void __box_rebuild_lambda(box_type *box, double a, double b, double h){
  int i,j,k;
  int pencil = box->pencil;
  int plane = box->plane;
  double h2inv = 1.0/(h*h);
  double * __restrict__ alpha  = box->grids[__alpha ];
  double * __restrict__ beta_i = box->grids[__beta_i];
  double * __restrict__ beta_j = box->grids[__beta_j];
  double * __restrict__ beta_k = box->grids[__beta_k];
  double * __restrict__ lambda = box->grids[__lambda];
  for(k=box->ghosts;k<box->dim_with_ghosts.k-box->ghosts;k++){
   for(j=box->ghosts;j<box->dim_with_ghosts.j-box->ghosts;j++){
    for(i=box->ghosts;i<box->dim_with_ghosts.i-box->ghosts;i++){
        int ijk = i + j*pencil + k*plane;
        double SumOfBetas = beta_i[ijk] + beta_i[ijk+     1] +
                            beta_j[ijk] + beta_j[ijk+pencil] +
                            beta_k[ijk] + beta_k[ijk+ plane];
        lambda[ijk] = 1.0 / (a*alpha[ijk] + b*SumOfBetas*h2inv);
  }}}
}


void __box_add_grids(box_type *box, int id_c, double scale_a, int id_a, double scale_b, int id_b){ // c=a+b
  int i,j,k;
  double * __restrict__ grid_c = box->grids[id_c];
  double * __restrict__ grid_a = box->grids[id_a];
  double * __restrict__ grid_b = box->grids[id_b];
  for(k=0;k<box->dim.k;k++){
   for(j=0;j<box->dim.j;j++){
    for(i=0;i<box->dim.i;i++){
      int ijk = (i+box->ghosts) + (j+box->ghosts)*box->pencil + (k+box->ghosts)*box->plane;
      grid_c[ijk] = scale_a*grid_a[ijk] + scale_b*grid_b[ijk];
  }}}
}


void __box_scale_grid(box_type *box, int id_c, double scale_a, int id_a){ // c=scale*a
  int i,j,k;
  double * __restrict__ grid_c = box->grids[id_c];
  double * __restrict__ grid_a = box->grids[id_a];
  for(k=0;k<box->dim.k;k++){
   for(j=0;j<box->dim.j;j++){
    for(i=0;i<box->dim.i;i++){
      int ijk = (i+box->ghosts) + (j+box->ghosts)*box->pencil + (k+box->ghosts)*box->plane;
      grid_c[ijk] = scale_a*grid_a[ijk];
  }}}
}


void __box_zero_grid(box_type *box,int grid_id){
  memset(box->grids[grid_id],0,box->volume*sizeof(double));
}

void __box_initialize_grid_to_scalar(box_type *box, int grid_id, double h, double scalar){
  int i,j,k;
  __box_zero_grid(box,grid_id);
  for(k=0;k<box->dim.k;k++){
  for(j=0;j<box->dim.j;j++){
  for(i=0;i<box->dim.i;i++){
    double x = h*(double)(i+box->low.i);
    double y = h*(double)(j+box->low.j);
    double z = h*(double)(k+box->low.k);
    int ijk = (i+box->ghosts) + (j+box->ghosts)*box->pencil + (k+box->ghosts)*box->plane;
    double value = (scalar);
    box->grids[grid_id][ijk] = value;
  }}}
}


//===========================================================================================
void zero_grid(domain_type * domain, int level, int grid_id){
  // zero's the entire grid including ghost zones...
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_zero_grid(&domain->subdomains[box].levels[level],grid_id);
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}

void initialize_grid_to_scalar(domain_type * domain, int level, int grid_id, double h, double scalar){
  // initializes the grid to a scalar while zero'ing the ghost zones...
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_initialize_grid_to_scalar(&domain->subdomains[box].levels[level],grid_id,h,scalar);
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}

void rebuild_lambda(domain_type * domain, int level, double a, double b, double hLevel){
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_rebuild_lambda(&domain->subdomains[box].levels[level],a,b,hLevel);
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}

void add_grids(domain_type * domain, int level, int id_c, double scale_a, int id_a, double scale_b, int id_b){ // c=a+b
  int box;
  uint64_t _timeStart = CycleTime();
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_add_grids(&domain->subdomains[box].levels[level],id_c,scale_a,id_a,scale_b,id_b);
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}

void scale_grid(domain_type * domain, int level, int id_c, double scale_a, int id_a){ // c=a+b
  int box;
  uint64_t _timeStart = CycleTime();
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_scale_grid(&domain->subdomains[box].levels[level],id_c,scale_a,id_a);
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}
