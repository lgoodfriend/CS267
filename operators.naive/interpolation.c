//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
void __box_interpolation(box_type *coarse, int id_c, box_type *fine, int id_f){ // coarse->fine
  int i,j,k;
  int pencil_f = fine->pencil;
  int  plane_f = fine->plane;
  double * __restrict__ grid_f    =   fine->grids[id_f];
  double * __restrict__ grid_c    = coarse->grids[id_c];
  for(k=0;k<coarse->dim.k;k++){
   for(j=0;j<coarse->dim.j;j++){
    for(i=0;i<coarse->dim.i;i++){
        int ijk_f = ((i<<1)+  fine->ghosts) + ((j<<1)+  fine->ghosts)*  fine->pencil + ((k<<1)+  fine->ghosts)*  fine->plane;
        int ijk_c = ((i   )+coarse->ghosts) + ((j   )+coarse->ghosts)*coarse->pencil + ((k   )+coarse->ghosts)*coarse->plane;
        grid_f[ijk_f                   ] += grid_c[ijk_c];
        grid_f[ijk_f                 +1] += grid_c[ijk_c];
        grid_f[ijk_f        +pencil_f  ] += grid_c[ijk_c];
        grid_f[ijk_f        +pencil_f+1] += grid_c[ijk_c];
        grid_f[ijk_f+plane_f           ] += grid_c[ijk_c];
        grid_f[ijk_f+plane_f         +1] += grid_c[ijk_c];
        grid_f[ijk_f+plane_f+pencil_f  ] += grid_c[ijk_c];
        grid_f[ijk_f+plane_f+pencil_f+1] += grid_c[ijk_c];
  }}}
}

void interpolation(domain_type * domain, int level_c, int id_c, int level_f, int id_f){
  int box;
  uint64_t _timeStart = CycleTime();
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_interpolation(&domain->subdomains[box].levels[level_c],id_c,&domain->subdomains[box].levels[level_f],id_f);
  }
  domain->cycles.interpolation[level_f] += (uint64_t)(CycleTime()-_timeStart);
}

