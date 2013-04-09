//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
void interpolation(domain_type * domain, int level_c, int id_c, int level_f, int id_f){
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in interpolation()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level_f].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level_f].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k;
    int ghosts_c = domain->subdomains[box].levels[level_c].ghosts;
    int pencil_c = domain->subdomains[box].levels[level_c].pencil;
    int  plane_c = domain->subdomains[box].levels[level_c].plane;
  
    int ghosts_f = domain->subdomains[box].levels[level_f].ghosts;
    int pencil_f = domain->subdomains[box].levels[level_f].pencil;
    int  plane_f = domain->subdomains[box].levels[level_f].plane;
    int  dim_i_f = domain->subdomains[box].levels[level_f].dim.i;
    int  dim_j_f = domain->subdomains[box].levels[level_f].dim.j;
    int  dim_k_f = domain->subdomains[box].levels[level_f].dim.k;
  
    double * __restrict__ grid_f = domain->subdomains[box].levels[level_f].grids[id_f] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f;
    double * __restrict__ grid_c = domain->subdomains[box].levels[level_c].grids[id_c] + ghosts_c*plane_c + ghosts_c*pencil_c + ghosts_c;
  
    #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
    for(k=0;k<dim_k_f;k++){
    for(j=0;j<dim_j_f;j++){
    for(i=0;i<dim_i_f;i++){
      int ijk_f = (i   ) + (j   )*pencil_f + (k   )*plane_f;
      int ijk_c = (i>>1) + (j>>1)*pencil_c + (k>>1)*plane_c;
      grid_f[ijk_f] += grid_c[ijk_c];
    }}}
  }
  domain->cycles.interpolation[level_f] += (uint64_t)(CycleTime()-_timeStart);
}

