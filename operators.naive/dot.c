//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
double __box_dot(box_type *box, int id_a, int id_b){
  double a_dot_b = 0.0;
  int i,j,k;
  double * __restrict__ grid_a = box->grids[id_a];
  double * __restrict__ grid_b = box->grids[id_b];
  for(k=0;k<box->dim.k;k++){
   for(j=0;j<box->dim.j;j++){
    for(i=0;i<box->dim.i;i++){
      int ijk = (i+box->ghosts) + (j+box->ghosts)*box->pencil + (k+box->ghosts)*box->plane;
      a_dot_b += grid_a[ijk]*grid_b[ijk];
  }}}
  return(a_dot_b);
}


double dot(domain_type * domain, int level, int id_a, int id_b){
  uint64_t _timeStart = CycleTime();
  double        a_dot_b =  0.0;
  int box;
  #pragma omp parallel for private(box)
  for(box=0;box<domain->numsubdomains;box++){
    double a_dot_b_box = __box_dot(&domain->subdomains[box].levels[level],id_a,id_b);
    #pragma omp critical
    {
      a_dot_b+=a_dot_b_box;
    }
  }
  domain->cycles.blas1[level] += (uint64_t)(CycleTime()-_timeStart);

  #ifdef _MPI
  uint64_t _timeStartAllReduce = CycleTime();
  double send = a_dot_b;
  MPI_Allreduce(&send,&a_dot_b,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  uint64_t _timeEndAllReduce = CycleTime();
  domain->cycles.collectives[level]   += (uint64_t)(_timeEndAllReduce-_timeStartAllReduce);
  domain->cycles.communication[level] += (uint64_t)(_timeEndAllReduce-_timeStartAllReduce);
  #endif

  return(a_dot_b);
}

