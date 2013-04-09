//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
void __box_smooth_GSRB_multiple(box_type *box, int phi_id, int rhs_id, double a, double b, double h, int sweep){
  int i,j,k,s;
  int pencil = box->pencil;
  int plane = box->plane;
  int ghosts = box->ghosts;
   double h2inv = 1.0/(h*h);
   double * __restrict__ phi    = box->grids[  phi_id] + ghosts*plane + ghosts*pencil + ghosts; // i.e. [0] = first non ghost zone point
   double * __restrict__ rhs    = box->grids[  rhs_id] + ghosts*plane + ghosts*pencil + ghosts;
   double * __restrict__ alpha  = box->grids[__alpha ] + ghosts*plane + ghosts*pencil + ghosts;
   double * __restrict__ beta_i = box->grids[__beta_i] + ghosts*plane + ghosts*pencil + ghosts;
   double * __restrict__ beta_j = box->grids[__beta_j] + ghosts*plane + ghosts*pencil + ghosts;
   double * __restrict__ beta_k = box->grids[__beta_k] + ghosts*plane + ghosts*pencil + ghosts;
   double * __restrict__ lambda = box->grids[__lambda] + ghosts*plane + ghosts*pencil + ghosts;
  uint64_t* __restrict__   mask = box->RedBlack_64bMask               + ghosts*pencil + ghosts;
  int color; //  0=red, 1=black
  int ghostsToOperateOn=ghosts-1;
  for(s=0,color=sweep;s<ghosts;s++,color++,ghostsToOperateOn--){
    for(k=0-ghostsToOperateOn;k<box->dim.k+ghostsToOperateOn;k++){
    for(j=0-ghostsToOperateOn;j<box->dim.j+ghostsToOperateOn;j++){
    for(i=0-ghostsToOperateOn;i<box->dim.i+ghostsToOperateOn;i++){
//    int ij = i+j*pencil;
//    if((mask[ij]^k^color^1)&1){
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
  }
  }}}}
}


void __box_smooth_GSRB_multiple_threaded(box_type *box, int phi_id, int rhs_id, double a, double b, double h, int sweep){
  int i,j,k,s;
  int pencil = box->pencil;
  int plane = box->plane;
  int ghosts = box->ghosts;
  double h2inv = 1.0/(h*h);
  double * __restrict__ phi    = box->grids[  phi_id] + ghosts*plane + ghosts*pencil + ghosts; // i.e. [0] = first non ghost zone point
  double * __restrict__ rhs    = box->grids[  rhs_id] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ alpha  = box->grids[__alpha ] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_i = box->grids[__beta_i] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_j = box->grids[__beta_j] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_k = box->grids[__beta_k] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ lambda = box->grids[__lambda] + ghosts*plane + ghosts*pencil + ghosts;
  int color; //  0=red, 1=black
  int ghostsToOperateOn=ghosts-1;
  for(s=0,color=sweep;s<ghosts;s++,color++,ghostsToOperateOn--){
    //#pragma omp parallel for private(k,j,i) collapse(2)
    #pragma omp parallel for private(k,j,i) 
    for(k=0-ghostsToOperateOn;k<box->dim.k+ghostsToOperateOn;k++){
    for(j=0-ghostsToOperateOn;j<box->dim.j+ghostsToOperateOn;j++){
    for(i=0-ghostsToOperateOn;i<box->dim.i+ghostsToOperateOn;i++){
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
  }
  }}}}
}

//------------------------------------------------------------------------------------------------------------------------------
void smooth(domain_type * domain, int level, int phi_id, int rhs_id, double a, double b, double hLevel, int sweep){
  uint64_t _timeStart = CycleTime();

  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  #ifdef __COLLABORATIVE_THREADING
    #warning using Collaborative Threading for large boxes...
    CollaborativeThreadingBoxSize = 1 << __COLLABORATIVE_THREADING;
  #endif

  int box;
  if(domain->subdomains[0].levels[level].dim.i >= CollaborativeThreadingBoxSize){
    for(box=0;box<domain->numsubdomains;box++){__box_smooth_GSRB_multiple_threaded(&domain->subdomains[box].levels[level],phi_id,rhs_id,a,b,hLevel,sweep);}
  }else{
    #pragma omp parallel for private(box)
    for(box=0;box<domain->numsubdomains;box++){__box_smooth_GSRB_multiple(&domain->subdomains[box].levels[level],phi_id,rhs_id,a,b,hLevel,sweep);}
  }
  domain->cycles.smooth[level] += (uint64_t)(CycleTime()-_timeStart);
}


