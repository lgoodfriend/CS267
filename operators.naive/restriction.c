//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
void __box_restriction(box_type *fine, int id_f, box_type *coarse, int id_c){ // fine->coarse
  int i,j,k;
  int pencil_f = fine->pencil;
  int plane_f = fine->plane;
  double * __restrict__ grid_f =   fine->grids[id_f];
  double * __restrict__ grid_c = coarse->grids[id_c];
  for(k=0;k<fine->dim.k;k+=2){
   for(j=0;j<fine->dim.j;j+=2){
    for(i=0;i<fine->dim.i;i+=2){
      int ijk_f = ((i   )+  fine->ghosts) + ((j   )+  fine->ghosts)*  fine->pencil + ((k   )+  fine->ghosts)*  fine->plane;
      int ijk_c = ((i>>1)+coarse->ghosts) + ((j>>1)+coarse->ghosts)*coarse->pencil + ((k>>1)+coarse->ghosts)*coarse->plane;
      grid_c[ijk_c] = ( grid_f[ijk_f                   ]+grid_f[ijk_f+1                 ] +
                        grid_f[ijk_f  +pencil_f        ]+grid_f[ijk_f+1+pencil_f        ] +
                        grid_f[ijk_f           +plane_f]+grid_f[ijk_f+1         +plane_f] +
                        grid_f[ijk_f  +pencil_f+plane_f]+grid_f[ijk_f+1+pencil_f+plane_f] ) * 0.125;
  }}}
}


void __box_restriction_betas(box_type *fine, box_type *coarse){
  int i,j,k,pencil_f,plane_f;
  double * __restrict__ beta_f;
  double * __restrict__ beta_c;

  // restrict beta_i 
  beta_f =   fine->grids[__beta_i];
  beta_c = coarse->grids[__beta_i];
  pencil_f = fine->pencil;
  plane_f = fine->plane;
  for(k=0;k<fine->dim.k;k+=2){
   for(j=0;j<fine->dim.j;j+=2){
    for(i=0;i<fine->dim.i;i+=2){
      int ijk_f = ((i   )+  fine->ghosts) + ((j   )+  fine->ghosts)*  fine->pencil + ((k   )+  fine->ghosts)*  fine->plane;
      int ijk_c = ((i>>1)+coarse->ghosts) + ((j>>1)+coarse->ghosts)*coarse->pencil + ((k>>1)+coarse->ghosts)*coarse->plane;
      beta_c[ijk_c] = ( beta_f[ijk_f        ]+beta_f[ijk_f+pencil_f        ] +
                        beta_f[ijk_f+plane_f]+beta_f[ijk_f+pencil_f+plane_f] ) * 0.25;
  }}}
  // restrict beta_j
  beta_f =   fine->grids[__beta_j];
  beta_c = coarse->grids[__beta_j];
  pencil_f = fine->pencil;
  plane_f = fine->plane;
  for(k=0;k<fine->dim.k;k+=2){
   for(j=0;j<fine->dim.j;j+=2){
    for(i=0;i<fine->dim.i;i+=2){
      int ijk_f = ((i   )+  fine->ghosts) + ((j   )+  fine->ghosts)*  fine->pencil + ((k   )+  fine->ghosts)*  fine->plane;
      int ijk_c = ((i>>1)+coarse->ghosts) + ((j>>1)+coarse->ghosts)*coarse->pencil + ((k>>1)+coarse->ghosts)*coarse->plane;
      beta_c[ijk_c] = ( beta_f[ijk_f        ]+beta_f[ijk_f+1        ] +
                        beta_f[ijk_f+plane_f]+beta_f[ijk_f+1+plane_f] ) * 0.25;
  }}}
  // restrict beta_k
  beta_f =   fine->grids[__beta_k];
  beta_c = coarse->grids[__beta_k];
  pencil_f = fine->pencil;
  plane_f = fine->plane;
  for(k=0;k<fine->dim.k;k+=2){
   for(j=0;j<fine->dim.j;j+=2){
    for(i=0;i<fine->dim.i;i+=2){
      int ijk_f = ((i   )+  fine->ghosts) + ((j   )+  fine->ghosts)*  fine->pencil + ((k   )+  fine->ghosts)*  fine->plane;
      int ijk_c = ((i>>1)+coarse->ghosts) + ((j>>1)+coarse->ghosts)*coarse->pencil + ((k>>1)+coarse->ghosts)*coarse->plane;
      beta_c[ijk_c] = ( beta_f[ijk_f         ]+beta_f[ijk_f+1         ] +
                        beta_f[ijk_f+pencil_f]+beta_f[ijk_f+1+pencil_f] ) * 0.25;
  }}}
}


//==========================================================================================================================
void restriction(domain_type * domain, int level_f, int id_f, int level_c, int id_c){
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_restriction(&domain->subdomains[box].levels[level_f],id_f,&domain->subdomains[box].levels[level_c],id_c);
  }
  domain->cycles.restriction[level_f] += (uint64_t)(CycleTime()-_timeStart);
}

void restriction_betas(domain_type * domain, int level_f, int level_c){
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_restriction_betas(&domain->subdomains[box].levels[level_f],&domain->subdomains[box].levels[level_c]);
  }
  domain->cycles.restriction[level_f] += (uint64_t)(CycleTime()-_timeStart);
}

