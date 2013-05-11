//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
double local_dot(domain_type * domain, int level, int id_a, int id_b){
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in dot()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);

  int box;
  double a_dot_b_domain =  0.0;
  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
    double * __restrict__ grid_a = domain->subdomains[box].levels[level].grids[id_a] + ghosts*plane + ghosts*pencil + ghosts; // i.e. [0] = first non ghost zone point
    double * __restrict__ grid_b = domain->subdomains[box].levels[level].grids[id_b] + ghosts*plane + ghosts*pencil + ghosts; 

    double a_dot_b_box = 0.0;
    #pragma omp parallel for private(i,j,k) if(omp_within_a_box) collapse(2) reduction(+:a_dot_b_box)
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
    for(i=0;i<dim_i;i++){
      int ijk = i + j*pencil + k*plane;
      a_dot_b_box += grid_a[ijk]*grid_b[ijk];
    }}}
    #pragma omp critical
    {
      a_dot_b_domain+=a_dot_b_box;
    }
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);

  return(a_dot_b_domain);
}

