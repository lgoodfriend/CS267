//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
/*
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
*/

//===========================================================================================
void zero_grid(domain_type * domain, int level, int grid_id){
/*
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_zero_grid(&domain->subdomains[box].levels[level],grid_id);
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
  uint64_t _timeStart = CycleTime();
*/

  // zero's the entire grid including ghost zones...
  uint64_t _timeStart = CycleTime();
  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in interpolation()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
    double * __restrict__ grid = domain->subdomains[box].levels[level].grids[grid_id];
    #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
    for(k=-ghosts;k<dim_k+ghosts;k++){
     for(j=-ghosts;j<dim_j+ghosts;j++){
      for(i=-ghosts;i<dim_i+ghosts;i++){
        int ijk = (i+ghosts) + (j+ghosts)*pencil + (k+ghosts)*plane;
        grid[ijk] = 0.0;
    }}}
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}



void initialize_grid_to_scalar(domain_type * domain, int level, int grid_id, double h, double scalar){
/*
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_initialize_grid_to_scalar(&domain->subdomains[box].levels[level],grid_id,h,scalar);
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
*/

  // initializes the grid to a scalar while zero'ing the ghost zones...
  uint64_t _timeStart = CycleTime();
  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in interpolation()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
    double * __restrict__ grid = domain->subdomains[box].levels[level].grids[grid_id];
    #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
    for(k=-ghosts;k<dim_k+ghosts;k++){
     for(j=-ghosts;j<dim_j+ghosts;j++){
      for(i=-ghosts;i<dim_i+ghosts;i++){
        int ijk = (i+ghosts) + (j+ghosts)*pencil + (k+ghosts)*plane;
        int ghostZone = (i<0) || (j<0) || (k<0) || (i>=dim_i) || (j>=dim_j) || (k>=dim_k);
        grid[ijk] = ghostZone ? 0.0 : scalar;
    }}}
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}



void rebuild_lambda(domain_type * domain, int level, double a, double b, double hLevel){
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in interpolation()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
    double h2inv = 1.0/(hLevel*hLevel);
    double * __restrict__ alpha  = domain->subdomains[box].levels[level].grids[__alpha ];
    double * __restrict__ beta_i = domain->subdomains[box].levels[level].grids[__beta_i];
    double * __restrict__ beta_j = domain->subdomains[box].levels[level].grids[__beta_j];
    double * __restrict__ beta_k = domain->subdomains[box].levels[level].grids[__beta_k];
    double * __restrict__ lambda = domain->subdomains[box].levels[level].grids[__lambda];
    #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
    for(k=0;k<dim_k;k++){
     for(j=0;j<dim_j;j++){
      for(i=0;i<dim_i;i++){
        int ijk = (i+ghosts) + (j+ghosts)*pencil + (k+ghosts)*plane;
        double SumOfBetas = beta_i[ijk] + beta_i[ijk+     1] +
                            beta_j[ijk] + beta_j[ijk+pencil] +
                            beta_k[ijk] + beta_k[ijk+ plane];
        lambda[ijk] = 1.0 / (a*alpha[ijk] + b*SumOfBetas*h2inv);
    }}}
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}



void add_grids(domain_type * domain, int level, int id_c, double scale_a, int id_a, double scale_b, int id_b){ // c=a+b
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in interpolation()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
    double * __restrict__ grid_c = domain->subdomains[box].levels[level].grids[id_c];
    double * __restrict__ grid_a = domain->subdomains[box].levels[level].grids[id_a];
    double * __restrict__ grid_b = domain->subdomains[box].levels[level].grids[id_b];
    #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
    for(k=0;k<dim_k;k++){
     for(j=0;j<dim_j;j++){
      for(i=0;i<dim_i;i++){
        int ijk = (i+ghosts) + (j+ghosts)*pencil + (k+ghosts)*plane;
        grid_c[ijk] = scale_a*grid_a[ijk] + scale_b*grid_b[ijk];
    }}}
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}



void scale_grid(domain_type * domain, int level, int id_c, double scale_a, int id_a){ // c=a+b
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in interpolation()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
    double * __restrict__ grid_c = domain->subdomains[box].levels[level].grids[id_c];
    double * __restrict__ grid_a = domain->subdomains[box].levels[level].grids[id_a];
    #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
    for(k=0;k<dim_k;k++){
     for(j=0;j<dim_j;j++){
      for(i=0;i<dim_i;i++){
        int ijk = (i+ghosts) + (j+ghosts)*pencil + (k+ghosts)*plane;
        grid_c[ijk] = scale_a*grid_a[ijk];
    }}}
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}


// a = [P,R]*bhat
void PR_mult(domain_type * domain, int level, int id_pstart, int pr_len, double *bhat, int id_a){ 
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in interpolation()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k,l;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
    double * __restrict__ grid_a  = domain->subdomains[box].levels[level].grids[id_a];
    /* TODO: Optimize this part */
     for(l=0 ; l < pr_len ; l++) {
      double * __restrict__ grid_pr = domain->subdomains[box].levels[level].grids[l+id_pstart];
      #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
      for(k=0;k<dim_k;k++){
       for(j=0;j<dim_j;j++){
        for(i=0;i<dim_i;i++){
          int ijk = (i+ghosts) + (j+ghosts)*pencil + (k+ghosts)*plane;
          if (l != 0) {
        	  grid_a[ijk] += bhat[l]*grid_pr[ijk];
          } else {
        	  grid_a[ijk] = bhat[l]*grid_pr[ijk];
          }

    }}}}
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);
}
