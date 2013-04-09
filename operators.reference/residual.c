//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
void __box_residual(box_type *box, int res_id, int phi_id, int rhs_id, double a, double b, double h){
  int i,j,k;
  int pencil = box->pencil;
  int plane = box->plane;
  int ghosts = box->ghosts;
  double h2inv = 1.0/(h*h);
  double * __restrict__ alpha  = box->grids[__alpha ] + ghosts*plane + ghosts*pencil + ghosts; // i.e. [0] = first non ghost zone point
  double * __restrict__ beta_i = box->grids[__beta_i] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_j = box->grids[__beta_j] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ beta_k = box->grids[__beta_k] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ lambda = box->grids[__lambda] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ phi    = box->grids[  phi_id] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ rhs    = box->grids[  rhs_id] + ghosts*plane + ghosts*pencil + ghosts;
  double * __restrict__ res    = box->grids[  res_id] + ghosts*plane + ghosts*pencil + ghosts;
  for(k=0;k<box->dim.k;k++){
   for(j=0;j<box->dim.j;j++){
    for(i=0;i<box->dim.i;i++){
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
        res[ijk] = rhs[ijk]-helmholtz;
  }}}
}


void __box_residual_and_restriction(box_type *fine, int phi_id, int rhs_id, box_type *coarse, int res_id, double a, double b, double h){
  int i,j,k;
  int ii,jj,kk;
  int pencil_f = fine->pencil;
  int  plane_f = fine->plane;
  int ghosts_f = fine->ghosts;
  int pencil_c = coarse->pencil;
  int  plane_c = coarse->plane;
  int ghosts_c = coarse->ghosts;
  double h2inv = 1.0/(h*h);
  double * __restrict__ alpha  =   fine->grids[__alpha ] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f; // i.e. [0] = first non ghost zone point
  double * __restrict__ beta_i =   fine->grids[__beta_i] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f;
  double * __restrict__ beta_j =   fine->grids[__beta_j] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f;
  double * __restrict__ beta_k =   fine->grids[__beta_k] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f;
  double * __restrict__ lambda =   fine->grids[__lambda] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f;
  double * __restrict__ phi    =   fine->grids[  phi_id] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f;
  double * __restrict__ rhs    =   fine->grids[  rhs_id] + ghosts_f*plane_f + ghosts_f*pencil_f + ghosts_f;
  double * __restrict__ res    = coarse->grids[  res_id] + ghosts_c*plane_c + ghosts_c*pencil_c + ghosts_c;
  for(k=0;k<fine->dim.k;k++){
    if( (k&1)==0 ){ // zero out the next plane for the coarse box (but only do it for every other 'fine' plane)
    for(j=0;j<coarse->dim.j;j++){
    for(i=0;i<coarse->dim.i;i++){
        int ijk_c = (i) + (j)*pencil_c + (k>>1)*plane_c;
        res[ijk_c] = 0.0;
    }}}
    for(j=0;j<fine->dim.j;j++){
    for(i=0;i<fine->dim.i;i++){
      int ijk_f = i + j*pencil_f + k*plane_f;
      double helmholtz =  a*alpha[ijk_f]*phi[ijk_f]
                         -b*h2inv*(
                            beta_i[ijk_f+1       ]*( phi[ijk_f+1       ]-phi[ijk_f         ] )
                           -beta_i[ijk_f         ]*( phi[ijk_f         ]-phi[ijk_f-1       ] )
                           +beta_j[ijk_f+pencil_f]*( phi[ijk_f+pencil_f]-phi[ijk_f         ] )
                           -beta_j[ijk_f         ]*( phi[ijk_f         ]-phi[ijk_f-pencil_f] )
                           +beta_k[ijk_f+plane_f ]*( phi[ijk_f+plane_f ]-phi[ijk_f         ] )
                           -beta_k[ijk_f         ]*( phi[ijk_f         ]-phi[ijk_f-plane_f ] )
                          );
      int ijk_c = (i>>1) + (j>>1)*pencil_c + (k>>1)*plane_c;
      res[ijk_c] += (rhs[ijk_f]-helmholtz)*0.125;
    }}
  }
}


//===========================================================================================================================
void residual(domain_type * domain, int level,  int res_id, int phi_id, int rhs_id, double a, double b, double hLevel){
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_residual(&domain->subdomains[box].levels[level],res_id,phi_id,rhs_id,a,b,hLevel);
  }
  domain->cycles.residual[level] += (uint64_t)(CycleTime()-_timeStart);
}

void residual_and_restriction(domain_type *domain, int level_f, int phi_id, int rhs_id, int level_c, int res_id, double a, double b, double hLevel){
  uint64_t _timeStart = CycleTime();
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    __box_residual_and_restriction(&domain->subdomains[box].levels[level_f],phi_id,rhs_id,&domain->subdomains[box].levels[level_c],res_id,a,b,hLevel);
  }
  domain->cycles.residual[level_f] += (uint64_t)(CycleTime()-_timeStart);
}
