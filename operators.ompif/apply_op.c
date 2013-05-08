//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
void apply_op(domain_type * domain, int level, int Ax_id, int x_id, double a, double b, double h, int deep ){  // y= Ax
  int CollaborativeThreadingBoxSize = 100000; // i.e. never
  //int deep = 1;
  #ifdef __COLLABORATIVE_THREADING
    //#warning using Collaborative Threading for large boxes in ApplyOp()
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
    double * __restrict__ x      = domain->subdomains[box].levels[level].grids[     x_id] + ghosts*plane + ghosts*pencil + ghosts; // i.e. [0] = first non ghost zone point
    double * __restrict__ Ax     = domain->subdomains[box].levels[level].grids[    Ax_id] + ghosts*plane + ghosts*pencil + ghosts; 
    double * __restrict__ alpha  = domain->subdomains[box].levels[level].grids[ __alpha ] + ghosts*plane + ghosts*pencil + ghosts;
    double * __restrict__ beta_i = domain->subdomains[box].levels[level].grids[ __beta_i] + ghosts*plane + ghosts*pencil + ghosts;
    double * __restrict__ beta_j = domain->subdomains[box].levels[level].grids[ __beta_j] + ghosts*plane + ghosts*pencil + ghosts;
    double * __restrict__ beta_k = domain->subdomains[box].levels[level].grids[ __beta_k] + ghosts*plane + ghosts*pencil + ghosts;

    #pragma omp parallel for private(k,j,i) if(omp_within_a_box) collapse(2)
    for(k=0-deep;k<dim_k+deep;k++){
    for(j=0-deep;j<dim_j+deep;j++){
    for(i=0-deep;i<dim_i+deep;i++){
      int ijk = i + j*pencil + k*plane;
      double helmholtz =  a*alpha[ijk]*x[ijk]
                         -b*h2inv*(
                            beta_i[ijk+1     ]*( x[ijk+1     ]-x[ijk       ] )
                           -beta_i[ijk       ]*( x[ijk       ]-x[ijk-1     ] )
                           +beta_j[ijk+pencil]*( x[ijk+pencil]-x[ijk       ] )
                           -beta_j[ijk       ]*( x[ijk       ]-x[ijk-pencil] )
                           +beta_k[ijk+plane ]*( x[ijk+plane ]-x[ijk       ] )
                           -beta_k[ijk       ]*( x[ijk       ]-x[ijk-plane ] )
                          );
      Ax[ijk] = helmholtz;
    }}}
  }
}

//------------------------------------------------------------------------------------------------------------------------------
