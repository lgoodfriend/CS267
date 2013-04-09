//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
void smooth(domain_type * domain, int level, int phi_id, int rhs_id, double a, double b, double h, int sweep){
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in smooth()
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif
  int omp_across_boxes = (domain->subdomains[0].levels[level].dim.i <  CollaborativeThreadingBoxSize);
  int omp_within_a_box = (domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize);
  int box;

  #pragma omp parallel for private(box) if(omp_across_boxes)
  for(box=0;box<domain->numsubdomains;box++){
    int i,j,k,s;
    int pencil = domain->subdomains[box].levels[level].pencil;
    int  plane = domain->subdomains[box].levels[level].plane;
    int ghosts = domain->subdomains[box].levels[level].ghosts;
    int  dim_k = domain->subdomains[box].levels[level].dim.k;
    int  dim_j = domain->subdomains[box].levels[level].dim.j;
    int  dim_i = domain->subdomains[box].levels[level].dim.i;
     double h2inv = 1.0/(h*h);
     double * __restrict__ phi    = domain->subdomains[box].levels[level].grids[  phi_id] + ghosts*plane + ghosts*pencil + ghosts; // i.e. [0] = first non ghost zone point
     double * __restrict__ rhs    = domain->subdomains[box].levels[level].grids[  rhs_id] + ghosts*plane + ghosts*pencil + ghosts;
     double * __restrict__ alpha  = domain->subdomains[box].levels[level].grids[__alpha ] + ghosts*plane + ghosts*pencil + ghosts;
     double * __restrict__ beta_i = domain->subdomains[box].levels[level].grids[__beta_i] + ghosts*plane + ghosts*pencil + ghosts;
     double * __restrict__ beta_j = domain->subdomains[box].levels[level].grids[__beta_j] + ghosts*plane + ghosts*pencil + ghosts;
     double * __restrict__ beta_k = domain->subdomains[box].levels[level].grids[__beta_k] + ghosts*plane + ghosts*pencil + ghosts;
     double * __restrict__ lambda = domain->subdomains[box].levels[level].grids[__lambda] + ghosts*plane + ghosts*pencil + ghosts;
    uint64_t* __restrict__   mask = domain->subdomains[box].levels[level].RedBlack_64bMask               + ghosts*pencil + ghosts;

    int color; //  0=red, 1=black
    int ghostsToOperateOn=ghosts-1;
    for(s=0,color=sweep;s<ghosts;s++,color++,ghostsToOperateOn--){
      #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
      for(k=0-ghostsToOperateOn;k<dim_k+ghostsToOperateOn;k++){
//      uint64_t   invertMask = 0-((k^color^)&1);
      for(j=0-ghostsToOperateOn;j<dim_j+ghostsToOperateOn;j++){
      for(i=0-ghostsToOperateOn;i<dim_i+ghostsToOperateOn;i++){
//      uint64_t    ColorMask = ( mask[ij])^invertMask;
        if((i^j^k^color^1)&1){ // looks very clean when [0] is i,j,k=0,0,0 
            int ijk = i + j*pencil + k*plane;
            double helmholtz =  a*alpha[ijk]*phi[ijk]
                               -b*h2inv*(
                                  beta_i[ijk+1     ]*( phi[ijk+1     ]-phi[ijk       ] )
                                 -beta_i[ijk       ]*( phi[ijk       ]-phi[ijk-1     ] )
                                 +beta_j[ijk+pencil]*( phi[ijk+pencil]-phi[ijk       ] )
                                 -beta_j[ijk       ]*( phi[ijk       ]-phi[ijk-pencil] )
                                 +beta_k[ijk+plane ]*( phi[ijk+plane ]-phi[ijk       ] )
                                 -beta_k[ijk       ]*( phi[ijk       ]-phi[ijk-plane ] )
                                );
            phi[ijk] = phi[ijk] - lambda[ijk]*(helmholtz-rhs[ijk]);
          //phi[ijk] = phi[ijk] - (ColorMask&(lambda[ijk]*(helmholtz-rhs[ijk]))); // bitwise and
          //phi[ijk] = phi[ijk] - mask[???][ij]*lambda[ijk]*(helmholtz-rhs[ijk]))); // mask[0|1][ij] is a set values in {1.0,0.0} 
        }
    }}}}
  }
  domain->cycles.smooth[level] += (uint64_t)(CycleTime()-_timeStart);
}

//------------------------------------------------------------------------------------------------------------------------------
