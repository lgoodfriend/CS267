//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
//#include <sched.h>
//------------------------------------------------------------------------------------------------------------------------------
#include <omp.h>
#ifdef _MPI
#include <mpi.h>
#endif
//------------------------------------------------------------------------------------------------------------------------------
#include "defines.h"
#include "box.h"
#include "mg.h"
#include "operators.h"
//==============================================================================
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#if defined(__x86_64__)
//  #warning x86/64 detected...
//#elif defined(__sparc) && defined (__sparcv9)
//  #warning Sparc detected
//#elif defined(__bgp__)
//  #warning BlueGene/P detected
//#else
//  #warning Defaulting to generic processor...
//#endif
////- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//#if defined(__CrayXT__)
//  #warning CrayXT detected...
////#include "../arch/generic/affinity.infoonly.c"
//#elif defined(__SOLARIS__)
//  #warning Solaris detected...
////#include "../arch/sparc/affinity.solaris.c"
//#elif defined(__bgp__)
//  #warning BlueGene/P detected...
////#include "../arch/generic/affinity.bgp.c"
//#else
//  #warning Defaulting to standard Linux cluster...
////#include "../arch/generic/affinity.reconstruct.c"
//#endif
//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



//==============================================================================
void __box_initialize_rhs(box_type *box, int grid_id, double h){
  int i,j,k;
  double twoPi = 2.0 * 3.1415926535;
  double value;
  memset(box->grids[grid_id],0,box->volume*sizeof(double)); // zero out the grid and ghost zones
  for(k=0;k<box->dim.k;k++){
  for(j=0;j<box->dim.j;j++){
  for(i=0;i<box->dim.i;i++){
    double x = h*(double)(i+box->low.i);
    double y = h*(double)(j+box->low.j);
    double z = h*(double)(k+box->low.k);
    int ijk = (i+box->ghosts) + (j+box->ghosts)*box->pencil + (k+box->ghosts)*box->plane;
    double value = sin(twoPi*x)*sin(twoPi*y)*sin(twoPi*z);
    /* XXX: This converges!
    if ((x>=0.25) && (x<0.75)){
      value = 1.0;
    }else{
      value = -1.0;
    }
    */
    /* XXX: This shouldn't converge!
    value = rand()*2. - 1.;
    */ 
    box->grids[grid_id][ijk] = value;
  }}}
}

//==============================================================================
void __box_check_answer(box_type *box, int grid_id, double h){
  int i,j,k;
  double twoPi = 2.0 * 3.1415926535;
  double solution;
  double pt_error;
  double eps = 0.00000001; // single precision
  double max_error = 0.0;

  for(k=0;k<box->dim.k;k++){
  for(j=0;j<box->dim.j;j++){
  for(i=0;i<box->dim.i;i++){
    double x = h*(double)(i+box->low.i);
    double y = h*(double)(j+box->low.j);
    double z = h*(double)(k+box->low.k);
    int ijk = (i+box->ghosts) + (j+box->ghosts)*box->pencil + (k+box->ghosts)*box->plane;
    double exact_solution = (-1.0/(3.0*twoPi*twoPi))*sin(twoPi*x)*sin(twoPi*y)*sin(twoPi*z);
    solution = box->grids[grid_id][ijk];
    pt_error = fabs(solution - exact_solution); // difference between exact solution and calculated solution
    if (pt_error > max_error){max_error = pt_error;} 
  }}}

  //if ( max_error > eps){printf("Maximum error in box = %16.8f \n",max_error);} 

} 


//==============================================================================
int main(int argc, char **argv){
  int MPI_Rank=0;
  int MPI_Tasks=1;
  int OMP_Threads = 1;

  #pragma omp parallel 
  {
    #pragma omp master
    {
      OMP_Threads = omp_get_num_threads();
    }
  }
    

  #ifdef _MPI
  #warning Compiling for MPI...
  int MPI_threadingModel          = -1;
//int MPI_threadingModelRequested = MPI_THREAD_SINGLE;
  int MPI_threadingModelRequested = MPI_THREAD_FUNNELED;
//int MPI_threadingModelRequested = MPI_THREAD_MULTIPLE;
  MPI_Init_thread(&argc, &argv, MPI_threadingModelRequested, &MPI_threadingModel);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_Tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_Rank);

  if(MPI_threadingModel>MPI_threadingModelRequested)MPI_threadingModel=MPI_threadingModelRequested;
  if(MPI_Rank==0){
       if(MPI_threadingModelRequested == MPI_THREAD_MULTIPLE  )printf("Requested MPI_THREAD_MULTIPLE, ");
  else if(MPI_threadingModelRequested == MPI_THREAD_SINGLE    )printf("Requested MPI_THREAD_SINGLE, ");
  else if(MPI_threadingModelRequested == MPI_THREAD_FUNNELED  )printf("Requested MPI_THREAD_FUNNELED, ");
  else if(MPI_threadingModelRequested == MPI_THREAD_SERIALIZED)printf("Requested MPI_THREAD_SERIALIZED, ");
  else if(MPI_threadingModelRequested == MPI_THREAD_MULTIPLE  )printf("Requested MPI_THREAD_MULTIPLE, ");
  else                                                printf("got Unknown MPI_threadingModel (%d)\n",MPI_threadingModel);
       if(MPI_threadingModel == MPI_THREAD_MULTIPLE  )printf("got MPI_THREAD_MULTIPLE\n");
  else if(MPI_threadingModel == MPI_THREAD_SINGLE    )printf("got MPI_THREAD_SINGLE\n");
  else if(MPI_threadingModel == MPI_THREAD_FUNNELED  )printf("got MPI_THREAD_FUNNELED\n");
  else if(MPI_threadingModel == MPI_THREAD_SERIALIZED)printf("got MPI_THREAD_SERIALIZED\n");
  else if(MPI_threadingModel == MPI_THREAD_MULTIPLE  )printf("got MPI_THREAD_MULTIPLE\n");
  else                                                printf("got Unknown MPI_threadingModel (%d)\n",MPI_threadingModel);
  fflush(stdout);  }
  #endif

//  timer_init();

  int log2_subdomain_dim = 6;
//    log2_subdomain_dim = 7;
//    log2_subdomain_dim = 5;
//    log2_subdomain_dim = 2;
  int subdomains_per_rank_in_i=256 / (1<<log2_subdomain_dim);
  int subdomains_per_rank_in_j=256 / (1<<log2_subdomain_dim);
  int subdomains_per_rank_in_k=256 / (1<<log2_subdomain_dim);
  int ranks_in_i=1;
  int ranks_in_j=1;
  int ranks_in_k=1;

  if(argc==2){
          log2_subdomain_dim=atoi(argv[1]);
          subdomains_per_rank_in_i=256 / (1<<log2_subdomain_dim);
          subdomains_per_rank_in_j=256 / (1<<log2_subdomain_dim);
          subdomains_per_rank_in_k=256 / (1<<log2_subdomain_dim);
  }else if(argc==5){
          log2_subdomain_dim=atoi(argv[1]);
    subdomains_per_rank_in_i=atoi(argv[2]);
    subdomains_per_rank_in_j=atoi(argv[3]);
    subdomains_per_rank_in_k=atoi(argv[4]);
  }else if(argc==8){
          log2_subdomain_dim=atoi(argv[1]);
    subdomains_per_rank_in_i=atoi(argv[2]);
    subdomains_per_rank_in_j=atoi(argv[3]);
    subdomains_per_rank_in_k=atoi(argv[4]);
                  ranks_in_i=atoi(argv[5]);
                  ranks_in_j=atoi(argv[6]);
                  ranks_in_k=atoi(argv[7]);
  }else if(argc!=1){
    if(MPI_Rank==0){printf("usage: ./a.out [log2_subdomain_dim]   [subdomains per rank in i,j,k]  [ranks in i,j,k]\n");}
    #ifdef _MPI
    MPI_Finalize();
    #endif
    exit(0);
  }

  if(log2_subdomain_dim>7){
    if(MPI_Rank==0){printf("error, log2_subdomain_dim(%d)>7\n",log2_subdomain_dim);}
    #ifdef _MPI
    MPI_Finalize();
    #endif
    exit(0);
  }

  if(ranks_in_i*ranks_in_j*ranks_in_k != MPI_Tasks){
    if(MPI_Rank==0){printf("error, ranks_in_i*ranks_in_j*ranks_in_k(%d*%d*%d=%d) != MPI_Tasks(%d)\n",ranks_in_i,ranks_in_j,ranks_in_k,ranks_in_i*ranks_in_j*ranks_in_k,MPI_Tasks);}
    #ifdef _MPI
    MPI_Finalize();
    #endif
    exit(0);
  }

  if(MPI_Rank==0)printf("%d MPI Tasks of %d threads\n",MPI_Tasks,OMP_Threads);

  int subdomain_dim_i=1<<log2_subdomain_dim;
  int subdomain_dim_j=1<<log2_subdomain_dim;
  int subdomain_dim_k=1<<log2_subdomain_dim;
  //    dim = 128 64 32 16 8 4
  // levels =   6  5  4  3 2 1
  int levels_in_vcycle=(log2_subdomain_dim+1)-2; // ie -log2(bottom size)

  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  domain_type domain_1 ;
  domain_type domain_CA;
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  create_domain(&domain_1 ,subdomain_dim_i,subdomain_dim_j,subdomain_dim_k,
                              subdomains_per_rank_in_i,subdomains_per_rank_in_j,subdomains_per_rank_in_k,
                              ranks_in_i,ranks_in_j,ranks_in_k,
                              MPI_Rank,10,1,levels_in_vcycle);
  create_domain(&domain_CA,subdomain_dim_i,subdomain_dim_j,subdomain_dim_k,
                              subdomains_per_rank_in_i,subdomains_per_rank_in_j,subdomains_per_rank_in_k,
                              ranks_in_i,ranks_in_j,ranks_in_k,
                              MPI_Rank,10,4,levels_in_vcycle);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  double  a=0.0;
  double  b=-1.0;
  double h0=1.0/((double)(domain_1.dim.i));
  int box;
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // define __alpha, __beta*, etc...
  initialize_grid_to_scalar( &domain_1,0,__alpha ,h0,1.0);
  initialize_grid_to_scalar( &domain_1,0,__beta_i,h0,1.0);
  initialize_grid_to_scalar( &domain_1,0,__beta_j,h0,1.0);
  initialize_grid_to_scalar( &domain_1,0,__beta_k,h0,1.0);
  initialize_grid_to_scalar(&domain_CA,0,__alpha ,h0,1.0);
  initialize_grid_to_scalar(&domain_CA,0,__beta_i,h0,1.0);
  initialize_grid_to_scalar(&domain_CA,0,__beta_j,h0,1.0);
  initialize_grid_to_scalar(&domain_CA,0,__beta_k,h0,1.0);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // define RHS
  for(box=0;box< domain_1.numsubdomains;box++){__box_initialize_rhs(& domain_1.subdomains[box].levels[0],__f,h0);}
  for(box=0;box<domain_CA.numsubdomains;box++){__box_initialize_rhs(&domain_CA.subdomains[box].levels[0],__f,h0);}
  // make initial guess for __u
  zero_grid(&domain_1 ,0,__u);
  zero_grid(&domain_CA,0,__u);
  //for(box=0;box< domain_1.numsubdomains;box++){__box_zero_grid(& domain_1.subdomains[box].levels[0],__u);}
  //for(box=0;box<domain_CA.numsubdomains;box++){__box_zero_grid(&domain_CA.subdomains[box].levels[0],__u);}
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  MGBuild(&domain_1 );
  MGBuild(&domain_CA);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  int s,sMax=2;
  #ifdef _MPI
  sMax=4;
  #endif
  for(s=0;s<sMax;s++)MGSolve(&domain_1 ,__u,__f,1,a,b,h0);
  for(s=0;s<sMax;s++)MGSolve(&domain_CA,__u,__f,1,a,b,h0);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // verification....
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  for(box=0;box< domain_1.numsubdomains;box++){__box_check_answer(& domain_1.subdomains[box].levels[0],__u,h0);}
  for(box=0;box<domain_CA.numsubdomains;box++){__box_check_answer(&domain_CA.subdomains[box].levels[0],__u,h0);}
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  print_timing(&domain_1 );
  print_timing(&domain_CA);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  destroy_domain(&domain_1 );
  destroy_domain(&domain_CA);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  #ifdef _MPI
  MPI_Finalize();
  #endif
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  return(0);
}
