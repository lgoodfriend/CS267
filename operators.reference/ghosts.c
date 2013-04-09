//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
inline void extract_from_grid(int low_i, int low_j, int low_k, int dim_i, int dim_j, int dim_k, int pencil, int plane, double * __restrict__ fromGrid, double * __restrict__ toBuf, int useCacheBypass){
  // extract a contiguous 3D array(toBuf) from coordinates low_i,j,k of fromGrid.  As fromGrid has a ghost zone, need pencil and plane sizes
  int i,j,k,b=0,ijk;
  if(dim_i==1){ // be smart and don't have an inner loop from 0 to 0
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
      int ijk = (  low_i) + (j+low_j)*pencil + (k+low_k)*plane;
      toBuf[b] = fromGrid[ijk];
      b++;
    }}
  }else{
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
    for(i=0;i<dim_i;i++){
      int ijk = (i+low_i) + (j+low_j)*pencil + (k+low_k)*plane;
      toBuf[b] = fromGrid[ijk];
      b++;
    }}}
  }
}


inline void insert_into_grid(int low_i, int low_j, int low_k, int dim_i, int dim_j, int dim_k, int pencil, int plane, double * __restrict__ fromBuf, double * __restrict__ toGrid, int useCacheBypass){
  // insert a contiguous 3D array(fromBuf) at coordinates low_i,j,k of toGrid.  As toGrid has a ghost zone, need pencil and plane sizes
  int i,j,k,b=0,ijk;
  if(dim_i==1){ // be smart and don't have an inner loop from 0 to 0
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
      int ijk = (  low_i) + (j+low_j)*pencil + (k+low_k)*plane;
      toGrid[ijk] = fromBuf[b];
      b++;
    }}
  }else{
    for(k=0;k<dim_k;k++){
    for(j=0;j<dim_j;j++){
    for(i=0;i<dim_i;i++){
      int ijk = (i+low_i) + (j+low_j)*pencil + (k+low_k)*plane;
      toGrid[ijk] = fromBuf[b];
      b++;
    }}}
  }
}


void __box_grid_to_surface_bufs(box_type *box, int grid_id){ // i.e. extract the (non ghost zone) surface of a box
  int ghosts = box->ghosts;
  int pencil = box->pencil;
  int plane  = box->plane;
  int dim_i = box->dim.i;
  int dim_j = box->dim.j;
  int dim_k = box->dim.k;

  int di,dj,dk;
  for(dk=-1;dk<=1;dk++){
  for(dj=-1;dj<=1;dj++){
  for(di=-1;di<=1;di++){int n=13+di+3*dj+9*dk;if(n!=13){
    int low_i,low_j,low_k;
    int buf_i,buf_j,buf_k;
    switch(di){
      case -1:low_i=ghosts;buf_i=ghosts;break;
      case  0:low_i=ghosts;buf_i= dim_i;break;
      case  1:low_i= dim_i;buf_i=ghosts;break;
    };
    switch(dj){
      case -1:low_j=ghosts;buf_j=ghosts;break;
      case  0:low_j=ghosts;buf_j= dim_j;break;
      case  1:low_j= dim_j;buf_j=ghosts;break;
    };
    switch(dk){
      case -1:low_k=ghosts;buf_k=ghosts;break;
      case  0:low_k=ghosts;buf_k= dim_k;break;
      case  1:low_k= dim_k;buf_k=ghosts;break;
    };
    extract_from_grid(low_i,low_j,low_k,buf_i,buf_j,buf_k,pencil,plane,box->grids[grid_id],box->surface_bufs[n],1);
  }}}}
}


void __box_ghost_bufs_to_grid(box_type *box, int grid_id){ // i.e. set the ghost zone values of a box
  int ghosts = box->ghosts;
  int pencil = box->pencil;
  int plane  = box->plane;
  int dim_i = box->dim.i;
  int dim_j = box->dim.j;
  int dim_k = box->dim.k;

  int di,dj,dk;
  for(dk=-1;dk<=1;dk++){
  for(dj=-1;dj<=1;dj++){
  for(di=-1;di<=1;di++){int n=13+di+3*dj+9*dk;if(n!=13){
    int low_i,low_j,low_k;
    int buf_i,buf_j,buf_k;
    switch(di){
      case -1:low_i=           0;buf_i=ghosts;break;
      case  0:low_i=      ghosts;buf_i= dim_i;break;
      case  1:low_i=ghosts+dim_i;buf_i=ghosts;break;
    };
    switch(dj){
      case -1:low_j=           0;buf_j=ghosts;break;
      case  0:low_j=      ghosts;buf_j= dim_j;break;
      case  1:low_j=ghosts+dim_j;buf_j=ghosts;break;
    };
    switch(dk){
      case -1:low_k=           0;buf_k=ghosts;break;
      case  0:low_k=      ghosts;buf_k= dim_k;break;
      case  1:low_k=ghosts+dim_k;buf_k=ghosts;break;
    };
    insert_into_grid(low_i,low_j,low_k,buf_i,buf_j,buf_k,pencil,plane,box->ghost_bufs[n],box->grids[grid_id],0);
  }}}}
}


