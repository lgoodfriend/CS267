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
#include "timer.h"
//------------------------------------------------------------------------------------------------------------------------------
int create_subdomain(subdomain_type * box, int subdomain_low_i, int subdomain_low_j, int subdomain_low_k,  
                                       int subdomain_dim_i, int subdomain_dim_j, int subdomain_dim_k, 
                                       int numGrids, int ghosts, int numLevels){
  int level;
  uint64_t memory_allocated=0;
  box->numLevels=numLevels;
  box->ghosts=ghosts;
  box->low.i = subdomain_low_i;
  box->low.j = subdomain_low_j;
  box->low.k = subdomain_low_k;
  box->dim.i = subdomain_dim_i;
  box->dim.j = subdomain_dim_j;
  box->dim.k = subdomain_dim_k;
  posix_memalign((void**)&(box->levels),64,numLevels*sizeof(box_type));
  memory_allocated += numLevels*sizeof(box_type);
  for(level=0;level<numLevels;level++){
    int __numGrids = numGrids;
    #ifdef __USE_BICGSTAB
    if(level == (numLevels-1))__numGrids+=6; // BiCGStab requires additional grids r0,r,p,s,Ap,As
    #endif
    memory_allocated += create_box(&box->levels[level],__numGrids,subdomain_low_i>>level,subdomain_low_j>>level,subdomain_low_k>>level,
                                                                  subdomain_dim_i>>level,subdomain_dim_j>>level,subdomain_dim_k>>level,ghosts);
  }
  return(memory_allocated);
}

void destroy_subdomain(subdomain_type * box){
  int level;
  for(level=0;level<box->numLevels;level++){
    destroy_box(&box->levels[level]);
  }
  free(box->levels);
}



//------------------------------------------------------------------------------------------------------------------------------
int calculate_neighboring_subdomain_index(int bi, int bj, int bk, int di, int dj, int dk, int subdomains_per_rank_in_i, int subdomains_per_rank_in_j, int subdomains_per_rank_in_k){
  int ni,nj,nk;
  ni=(bi+subdomains_per_rank_in_i+di)%subdomains_per_rank_in_i;
  nj=(bj+subdomains_per_rank_in_j+dj)%subdomains_per_rank_in_j;
  nk=(bk+subdomains_per_rank_in_k+dk)%subdomains_per_rank_in_k;
  int index = ni + nj*subdomains_per_rank_in_i + nk*subdomains_per_rank_in_i*subdomains_per_rank_in_j;
  return(index);
}

int calculate_neighboring_subdomain_rank(int bi, int bj, int bk, int di, int dj, int dk, int subdomains_per_rank_in_i, int subdomains_per_rank_in_j, int subdomains_per_rank_in_k,
                                   int ri, int rj, int rk, int ranks_in_i, int ranks_in_j, int ranks_in_k){
  if((bi+di)<0)ri--;if((bi+di)>=subdomains_per_rank_in_i)ri++;ri=(ri+ranks_in_i)%ranks_in_i;
  if((bj+dj)<0)rj--;if((bj+dj)>=subdomains_per_rank_in_j)rj++;rj=(rj+ranks_in_j)%ranks_in_j;
  if((bk+dk)<0)rk--;if((bk+dk)>=subdomains_per_rank_in_k)rk++;rk=(rk+ranks_in_k)%ranks_in_k;
  int rank = ri + rj*ranks_in_i + rk*ranks_in_i*ranks_in_j;
  return(rank);
}

int on_same_face(int n1, int n2){
  int i1 = ((n1  )%3)-1;
  int j1 = ((n1/3)%3)-1;
  int k1 = ((n1/9)%3)-1;
  int i2 = ((n2  )%3)-1;
  int j2 = ((n2/3)%3)-1;
  int k2 = ((n2/9)%3)-1;
  if( (i1!=0)&&(i1==i2) )return(1);
  if( (j1!=0)&&(j1==j2) )return(1);
  if( (k1!=0)&&(k1==k2) )return(1);
  return(0);
}

int on_same_edge(int n1, int n2){
  int i1 = ((n1  )%3)-1;
  int j1 = ((n1/3)%3)-1;
  int k1 = ((n1/9)%3)-1;
  int i2 = ((n2  )%3)-1;
  int j2 = ((n2/3)%3)-1;
  int k2 = ((n2/9)%3)-1;
  if( (i1!=0) && (j1!=0) && (i1==i2) && (j1==j2) )return(1);
  if( (i1!=0) && (k1!=0) && (i1==i2) && (k1==k2) )return(1);
  if( (j1!=0) && (k1!=0) && (j1==j2) && (k1==k2) )return(1);
  return(0);
}
//------------------------------------------------------------------------------------------------------------------------------
int create_domain(domain_type * domain, 
              int subdomain_dim_i,  int subdomain_dim_j,  int subdomain_dim_k,  
              int subdomains_per_rank_in_i, int subdomains_per_rank_in_j, int subdomains_per_rank_in_k, 
              int ranks_in_i,      int ranks_in_j,      int ranks_in_k, 
              int rank,
              int numGrids, int ghosts, int numLevels
             ){
  int  i, j, k;
  int di,dj,dk;

  domain->rank = rank;
  if(domain->rank==0){printf("creating domain...       ");fflush(stdout);}
  if(ghosts>subdomain_dim_i>>(numLevels-1)){if(domain->rank==0)printf("#ghosts(%d)>bottom grid size(%d)\n",ghosts,subdomain_dim_i>>(numLevels-1));exit(0);}

  if( (subdomain_dim_i!=subdomain_dim_j)||(subdomain_dim_j!=subdomain_dim_k)||(subdomain_dim_i!=subdomain_dim_k) ){
  if(domain->rank==0)printf("subdomain_dim's must be equal\n");exit(0);
  }
  uint64_t memory_allocated =0;
  // processes are laid out in x, then y, then z
  //int ranks = ranks_in_i*ranks_in_j*ranks_in_k;
  int my_k_rank = (rank                                                              ) / (ranks_in_i*ranks_in_j);
  int my_j_rank = (rank - (ranks_in_i*ranks_in_j*my_k_rank)                          ) / (ranks_in_i           );
  int my_i_rank = (rank - (ranks_in_i*ranks_in_j*my_k_rank) - (ranks_in_i*my_j_rank) );
  //printf("%2d: (%2d,%2d,%2d)\n",domain->rank,my_k_rank,my_j_rank,my_i_rank);

  for(dk=-1;dk<=1;dk++){
  for(dj=-1;dj<=1;dj++){
  for(di=-1;di<=1;di++){
    int n = 13+di+3*dj+9*dk;
    int neighbor_rank_in_i = (my_i_rank+di+ranks_in_i)%ranks_in_i;
    int neighbor_rank_in_j = (my_j_rank+dj+ranks_in_j)%ranks_in_j;
    int neighbor_rank_in_k = (my_k_rank+dk+ranks_in_k)%ranks_in_k;
    domain->rank_of_neighbor[n] = neighbor_rank_in_i + ranks_in_i*neighbor_rank_in_j + ranks_in_i*ranks_in_j*neighbor_rank_in_k;
  }}}

  domain->subdomains_per_rank_in.i = subdomains_per_rank_in_i;
  domain->subdomains_per_rank_in.j = subdomains_per_rank_in_j;
  domain->subdomains_per_rank_in.k = subdomains_per_rank_in_k;
  domain->numsubdomains = subdomains_per_rank_in_i*subdomains_per_rank_in_j*subdomains_per_rank_in_k;
  posix_memalign((void**)&(domain->subdomains),64,domain->numsubdomains*sizeof(subdomain_type));
  memory_allocated+=domain->numsubdomains*sizeof(subdomain_type);

  domain->subdomains_in.i = domain->subdomains_per_rank_in.i * ranks_in_i;
  domain->subdomains_in.j = domain->subdomains_per_rank_in.j * ranks_in_j;
  domain->subdomains_in.k = domain->subdomains_per_rank_in.k * ranks_in_k;
  domain->dim.i = domain->subdomains_in.i * subdomain_dim_i;
  domain->dim.j = domain->subdomains_in.j * subdomain_dim_j;
  domain->dim.k = domain->subdomains_in.k * subdomain_dim_k;
  domain->numLevels = numLevels;
  domain->numGrids  = numGrids;
  domain->ghosts = ghosts;

  // subdomains within a process are laid out in i, then j, then k
  for(k=0;k<subdomains_per_rank_in_k;k++){
  for(j=0;j<subdomains_per_rank_in_j;j++){
  for(i=0;i<subdomains_per_rank_in_i;i++){
    int box = i + j*subdomains_per_rank_in_i + k*subdomains_per_rank_in_i*subdomains_per_rank_in_j;
    int low_i = subdomain_dim_i * (i + subdomains_per_rank_in_i*my_i_rank);
    int low_j = subdomain_dim_j * (j + subdomains_per_rank_in_j*my_j_rank);
    int low_k = subdomain_dim_k * (k + subdomains_per_rank_in_k*my_k_rank);
    memory_allocated += create_subdomain(&domain->subdomains[box],low_i,low_j,low_k,
                                         //(i+subdomains_per_rank_in_i)*subdomain_dim_i,(j+subdomains_per_rank_in_j)*subdomain_dim_j,(k+subdomains_per_rank_in_k)*subdomain_dim_k,
                                            subdomain_dim_i,                             subdomain_dim_j,                             subdomain_dim_k,
                                            numGrids,ghosts,numLevels);
    for(dk=-1;dk<=1;dk++){
    for(dj=-1;dj<=1;dj++){
    for(di=-1;di<=1;di++){
      int n = 13+di+3*dj+9*dk;
    domain->subdomains[box].neighbors[n].rank        = calculate_neighboring_subdomain_rank( i,j,k,di,dj,dk, subdomains_per_rank_in_i,subdomains_per_rank_in_j,subdomains_per_rank_in_k,
                                                                                                               my_i_rank,my_j_rank,my_k_rank,ranks_in_i,ranks_in_j,ranks_in_k);
    domain->subdomains[box].neighbors[n].local_index = calculate_neighboring_subdomain_index(i,j,k,di,dj,dk, subdomains_per_rank_in_i,subdomains_per_rank_in_j,subdomains_per_rank_in_k);
      //printf("rank=%2d, box[%2d].neighbors[%3d]=(%3d,%3d)\n",domain->rank,box,n,domain->subdomains[box].neighbors[n].rank,domain->subdomains[box].neighbors[n].local_index);
    }}}
  }}}


  #ifdef _MPI
  int   FaceSizeAtLevel0 = subdomain_dim_i*subdomain_dim_i*ghosts;
  int   EdgeSizeAtLevel0 = subdomain_dim_i*ghosts*ghosts;
  int CornerSizeAtLevel0 = ghosts*ghosts*ghosts;






  int   FaceBufSizePerSubdomain = FaceSizeAtLevel0 + 4*EdgeSizeAtLevel0 + 4*CornerSizeAtLevel0;
  int   EdgeBufSizePerSubdomain =                      EdgeSizeAtLevel0 + 2*CornerSizeAtLevel0;
  int CornerBufSizePerSubdomain =                                           CornerSizeAtLevel0;

  int    faces[27] = {0,0,0,0,1,0,0,0,0,  0,1,0,1,0,1,0,1,0,  0,0,0,0,1,0,0,0,0};
  int    edges[27] = {0,1,0,1,0,1,0,1,0,  1,0,1,0,0,0,1,0,1,  0,1,0,1,0,1,0,1,0};
  int  corners[27] = {1,0,1,0,0,0,1,0,1,  0,0,0,0,0,0,0,0,0,  1,0,1,0,0,0,1,0,1};

  // allocate MPI send/recv buffers for the 26 neighbors....
  for(dk=-1;dk<=1;dk++){
  for(dj=-1;dj<=1;dj++){
  for(di=-1;di<=1;di++){
    int n = 13+di+3*dj+9*dk;
    if(faces[n]){
      int SubdomainsWritingToThisBuffer;
      if(di!=0)SubdomainsWritingToThisBuffer=domain->subdomains_per_rank_in.j*domain->subdomains_per_rank_in.k;
      if(dj!=0)SubdomainsWritingToThisBuffer=domain->subdomains_per_rank_in.i*domain->subdomains_per_rank_in.k;
      if(dk!=0)SubdomainsWritingToThisBuffer=domain->subdomains_per_rank_in.i*domain->subdomains_per_rank_in.j;
      posix_memalign((void**)&(domain->send_buffer[n]),64,FaceBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
      posix_memalign((void**)&(domain->recv_buffer[n]),64,FaceBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                          memset(domain->send_buffer[n],0,FaceBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                          memset(domain->recv_buffer[n],0,FaceBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                                        memory_allocated+=FaceBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double);
                                        memory_allocated+=FaceBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double);
      domain->buffer_size[n].faces   = 1*SubdomainsWritingToThisBuffer;
      domain->buffer_size[n].edges   = 4*SubdomainsWritingToThisBuffer;
      domain->buffer_size[n].corners = 4*SubdomainsWritingToThisBuffer;
    }
    if(edges[n]){
      int SubdomainsWritingToThisBuffer;
      if(di==0)SubdomainsWritingToThisBuffer=domain->subdomains_per_rank_in.i;
      if(dj==0)SubdomainsWritingToThisBuffer=domain->subdomains_per_rank_in.j;
      if(dk==0)SubdomainsWritingToThisBuffer=domain->subdomains_per_rank_in.k;
      posix_memalign((void**)&(domain->send_buffer[n]),64,EdgeBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
      posix_memalign((void**)&(domain->recv_buffer[n]),64,EdgeBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                          memset(domain->send_buffer[n],0,EdgeBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                          memset(domain->recv_buffer[n],0,EdgeBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                                        memory_allocated+=EdgeBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double);
                                        memory_allocated+=EdgeBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double);
      domain->buffer_size[n].faces   = 0;
      domain->buffer_size[n].edges   = 1*SubdomainsWritingToThisBuffer;
      domain->buffer_size[n].corners = 2*SubdomainsWritingToThisBuffer;
    }
    if(corners[n]){
      int SubdomainsWritingToThisBuffer=1;
      posix_memalign((void**)&(domain->send_buffer[n]),64,CornerBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
      posix_memalign((void**)&(domain->recv_buffer[n]),64,CornerBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                          memset(domain->send_buffer[n],0,CornerBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                          memset(domain->recv_buffer[n],0,CornerBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double));
                                        memory_allocated+=CornerBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double);
                                        memory_allocated+=CornerBufSizePerSubdomain*SubdomainsWritingToThisBuffer*sizeof(double);
      domain->buffer_size[n].faces   = 0;
      domain->buffer_size[n].edges   = 0;
      domain->buffer_size[n].corners = 1;
    }
  }}}

  //if(domain->rank==0)printf("\n");
  struct{int faces,edges,corners;}send_offset[27][28]; // offset[buffer][neighbor]... ie location of neighbor
  struct{int faces,edges,corners;}recv_offset[27][28]; // offset[buffer][neighbor]... ie location of neighbor

  int n1,n2,n3;
  for(n1=0;n1<27;n1++){
      send_offset[n1][27].faces  =0;
      send_offset[n1][27].edges  =0;
      send_offset[n1][27].corners=0;
      recv_offset[n1][27].faces  =0;
      recv_offset[n1][27].edges  =0;
      recv_offset[n1][27].corners=0;
    for(n2=0;n2<27;n2++){
      n3 = 26-n2;
      send_offset[n1][n2].faces  =0;
      send_offset[n1][n2].edges  =0;
      send_offset[n1][n2].corners=0;
      recv_offset[n1][n3].faces  =0;
      recv_offset[n1][n3].edges  =0;
      recv_offset[n1][n3].corners=0;
      if( (faces[n1] && on_same_face(n1,n2)) || (edges[n1] && on_same_edge(n1,n2)) || (corners[n1] && (n1==n2)) ){
        //if(domain->rank==0)printf("%d (%d,%d) (%d,%d) (%d,%d)\n",n2,faces[n1],on_same_face(n1,n2),edges[n1],on_same_edge(n1,n2),corners[n1],(n1==n2));
        send_offset[n1][n2].faces  =send_offset[n1][27].faces;  send_offset[n1][27].faces  +=  faces[n2];
        send_offset[n1][n2].edges  =send_offset[n1][27].edges;  send_offset[n1][27].edges  +=  edges[n2];
        send_offset[n1][n2].corners=send_offset[n1][27].corners;send_offset[n1][27].corners+=corners[n2];
      }
      if( (faces[n1] && on_same_face(n1,n3)) || (edges[n1] && on_same_edge(n1,n3)) || (corners[n1] && (n1==n3)) ){
        //if(domain->rank==0)printf("%d (%d,%d) (%d,%d) (%d,%d)\n",n3,faces[n1],on_same_face(n1,n3),edges[n1],on_same_edge(n1,n3),corners[n1],(n1==n3));
        recv_offset[n1][n3].faces  =recv_offset[n1][27].faces;  recv_offset[n1][27].faces  +=  faces[n3];
        recv_offset[n1][n3].edges  =recv_offset[n1][27].edges;  recv_offset[n1][27].edges  +=  edges[n3];
        recv_offset[n1][n3].corners=recv_offset[n1][27].corners;recv_offset[n1][27].corners+=corners[n3];
      }
    }
    //if(domain->rank==0)printf("%2d: ",n1);
    //for(n2=0;n2<27;n2++){
    //  if(domain->rank==0)printf("(%d,%d,%d) ",recv_offset[n1][n2].faces,recv_offset[n1][n2].edges,recv_offset[n1][n2].corners);
    //}
    //if(domain->rank==0)printf("\n");
  }

  // for all ijk
  // for all dijk
  for(k=0;k<subdomains_per_rank_in_k;k++){
  for(j=0;j<subdomains_per_rank_in_j;j++){
  for(i=0;i<subdomains_per_rank_in_i;i++){
    int box = i + j*subdomains_per_rank_in_i + k*subdomains_per_rank_in_i*subdomains_per_rank_in_j;
    for(dk=-1;dk<=1;dk++){
    for(dj=-1;dj<=1;dj++){
    for(di=-1;di<=1;di++){
      int n = 13+di+3*dj+9*dk;
         domain->subdomains[box].neighbors[n].send.buf = -1;
         domain->subdomains[box].neighbors[n].recv.buf = -1;
      if(domain->subdomains[box].neighbors[n].rank != domain->rank){
        int ni = i+di;
        int nj = j+dj;
        int nk = k+dk;
        int buf_di=0;if(ni<0)buf_di=-1;if(ni>=subdomains_per_rank_in_i)buf_di=1;
        int buf_dj=0;if(nj<0)buf_dj=-1;if(nj>=subdomains_per_rank_in_j)buf_dj=1;
        int buf_dk=0;if(nk<0)buf_dk=-1;if(nk>=subdomains_per_rank_in_k)buf_dk=1;
        // now decide where this box writes surface data to and where it reads ghost data from
        int buf;
        int base=-1000;
        buf = 13+buf_di+3*buf_dj+9*buf_dk;
        if(faces[buf]){ 
          // send
          if(buf_di!=0){base = (j+k*subdomains_per_rank_in_j);}// jk plane
          if(buf_dj!=0){base = (i+k*subdomains_per_rank_in_i);}// ik plane
          if(buf_dk!=0){base = (i+j*subdomains_per_rank_in_i);}// ij plane
          domain->subdomains[box].neighbors[n].send.buf = buf;
          domain->subdomains[box].neighbors[n].send.offset.faces   = 1*base+send_offset[buf][n].faces;
          domain->subdomains[box].neighbors[n].send.offset.edges   = 4*base+send_offset[buf][n].edges;
          domain->subdomains[box].neighbors[n].send.offset.corners = 4*base+send_offset[buf][n].corners;
          // recv
          if(buf_di!=0){base = (nj+nk*subdomains_per_rank_in_j);}// jk plane
          if(buf_dj!=0){base = (ni+nk*subdomains_per_rank_in_i);}// ik plane
          if(buf_dk!=0){base = (ni+nj*subdomains_per_rank_in_i);}// ij plane
          domain->subdomains[box].neighbors[n].recv.buf = buf;
          domain->subdomains[box].neighbors[n].recv.offset.faces   = 1*base+recv_offset[buf][n].faces;
          domain->subdomains[box].neighbors[n].recv.offset.edges   = 4*base+recv_offset[buf][n].edges;
          domain->subdomains[box].neighbors[n].recv.offset.corners = 4*base+recv_offset[buf][n].corners;
        }
        if(edges[buf]){ 
          // send
          if(buf_di==0){base = i;} // edge along i
          if(buf_dj==0){base = j;} // edge along j
          if(buf_dk==0){base = k;} // edge along k
          domain->subdomains[box].neighbors[n].send.buf = buf;
          domain->subdomains[box].neighbors[n].send.offset.faces   = 0*base+send_offset[buf][n].faces;
          domain->subdomains[box].neighbors[n].send.offset.edges   = 1*base+send_offset[buf][n].edges;
          domain->subdomains[box].neighbors[n].send.offset.corners = 2*base+send_offset[buf][n].corners;
          // recv
          if(buf_di==0){base = ni;} // edge along i
          if(buf_dj==0){base = nj;} // edge along j
          if(buf_dk==0){base = nk;} // edge along k
          domain->subdomains[box].neighbors[n].recv.buf = buf;
          domain->subdomains[box].neighbors[n].recv.offset.faces   = 0*base+recv_offset[buf][n].faces;
          domain->subdomains[box].neighbors[n].recv.offset.edges   = 1*base+recv_offset[buf][n].edges;
          domain->subdomains[box].neighbors[n].recv.offset.corners = 2*base+recv_offset[buf][n].corners;
        }
        if(corners[buf]){
          // send
          domain->subdomains[box].neighbors[n].send.buf = buf;
          domain->subdomains[box].neighbors[n].send.offset.faces   = send_offset[buf][n].faces;
          domain->subdomains[box].neighbors[n].send.offset.edges   = send_offset[buf][n].edges;
          domain->subdomains[box].neighbors[n].send.offset.corners = send_offset[buf][n].corners;
          // recv
          domain->subdomains[box].neighbors[n].recv.buf = buf;
          domain->subdomains[box].neighbors[n].recv.offset.faces   = recv_offset[buf][n].faces;
          domain->subdomains[box].neighbors[n].recv.offset.edges   = recv_offset[buf][n].edges;
          domain->subdomains[box].neighbors[n].recv.offset.corners = recv_offset[buf][n].corners;
        }
  
        //if(domain->rank==0)printf("box=%2d, n=%2d: send.buf=%2d, send.offset={%2d,%2d,%2d}\n",box,n,
        //  domain->subdomains[box].neighbors[n].send.buf,
        //  domain->subdomains[box].neighbors[n].send.offset.faces,   
        //  domain->subdomains[box].neighbors[n].send.offset.edges,  
        //  domain->subdomains[box].neighbors[n].send.offset.corners);
        //if(domain->rank==0)printf("box=%2d, n=%2d: recv.buf=%2d, recv.offset={%2d,%2d,%2d} ijk=(%2d,%2d,%2d), nijk=(%2d,%2d,%2d)\n",box,n,
        //  domain->subdomains[box].neighbors[n].recv.buf,
        //  domain->subdomains[box].neighbors[n].recv.offset.faces,   
        //  domain->subdomains[box].neighbors[n].recv.offset.edges,  
        //  domain->subdomains[box].neighbors[n].recv.offset.corners,i,j,k,ni,nj,nk);

      }
    }}}
  }}}
  #endif


  if(domain->rank==0){
  printf("done\n");fflush(stdout); 
  printf("  %d x %d x %d (per subdomain)\n",subdomain_dim_i,subdomain_dim_j,subdomain_dim_k);
  printf("  %d x %d x %d (per process)\n",subdomains_per_rank_in_i*subdomain_dim_i,subdomains_per_rank_in_j*subdomain_dim_j,subdomains_per_rank_in_k*subdomain_dim_k);
  printf("  %d x %d x %d (overall)\n",ranks_in_i*subdomains_per_rank_in_i*subdomain_dim_i,ranks_in_j*subdomains_per_rank_in_j*subdomain_dim_j,ranks_in_k*subdomains_per_rank_in_k*subdomain_dim_k);
  printf("  %d-deep ghost zones\n",ghosts);
  printf("  allocated %llu MB\n",memory_allocated>>20);
  fflush(stdout);
  }
  return(memory_allocated);
}

void destroy_domain(domain_type * domain){
  if(domain->rank==0){printf("deallocating domain...   ");fflush(stdout);}
  int box;for(box=0;box<domain->numsubdomains;box++){
    destroy_subdomain(&domain->subdomains[box]);
  }
  free(domain->subdomains);
  if(domain->rank==0){printf("done\n");fflush(stdout);}
}

//------------------------------------------------------------------------------------------------------------------------------
void print_timing(domain_type *domain){
  int level,numLevels = domain->numLevels;
  uint64_t _timeStart=CycleTime();sleep(1);uint64_t _timeEnd=CycleTime();
  double SecondsPerCycle = (double)1.0/(double)(_timeEnd-_timeStart);

  if(domain->rank!=0)return;

  uint64_t total;
          printf("                       ");for(level=0;level<(numLevels  );level++){printf("%12d ",level);}printf("\n");
          printf("                       ");for(level=0;level<(numLevels  );level++){printf("%10d^3 ",domain->subdomains[0].levels[0].dim.i>>level);}printf("       total\n");
  total=0;printf("smooth                 ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)       domain->cycles.smooth[level]);total+=       domain->cycles.smooth[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("residual               ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)     domain->cycles.residual[level]);total+=     domain->cycles.residual[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("restriction            ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)  domain->cycles.restriction[level]);total+=  domain->cycles.restriction[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("interpolation          ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)domain->cycles.interpolation[level]);total+=domain->cycles.interpolation[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
#ifdef __PRINT_NORM
  total=0;printf("norm                   ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)         domain->cycles.norm[level]);total+=         domain->cycles.norm[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
#endif
  total=0;printf("BLAS1                  ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)        domain->cycles.blas1[level]);total+=        domain->cycles.blas1[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("communication          ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)domain->cycles.communication[level]);total+=domain->cycles.communication[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
#ifdef __PRINT_COMMUNICATION_BREAKDOWN
  total=0;printf("  surface to buffers   ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)        domain->cycles.s2buf[level]);total+=        domain->cycles.s2buf[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("  exchange buffers     ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)      domain->cycles.bufcopy[level]);total+=      domain->cycles.bufcopy[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("  buffers to ghosts    ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)        domain->cycles.buf2g[level]);total+=        domain->cycles.buf2g[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  #ifdef _MPI
  total=0;printf("  pack MPI buffers     ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)         domain->cycles.pack[level]);total+=         domain->cycles.pack[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("  unpack MPI buffers   ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)         domain->cycles.unpack[level]);total+=         domain->cycles.unpack[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("  MPI_Isend            ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)         domain->cycles.send[level]);total+=         domain->cycles.send[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("  MPI_Irecv            ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)         domain->cycles.recv[level]);total+=         domain->cycles.recv[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("  MPI_Waitall          ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)         domain->cycles.wait[level]);total+=         domain->cycles.wait[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  total=0;printf("  MPI_collectives      ");for(level=0;level<(numLevels  );level++){printf("%12.6f ",SecondsPerCycle*(double)  domain->cycles.collectives[level]);total+=  domain->cycles.collectives[level];}printf("%12.6f\n",SecondsPerCycle*(double)total);
  #endif
#endif
  total=0;printf("--------------         ");for(level=0;level<(numLevels+1);level++){printf("------------ ");}printf("\n");
  total=0;printf("Total by level         ");for(level=0;level<(numLevels  );level++){printf("%12.6f " ,SecondsPerCycle*(double)domain->cycles.Total[level]);total+=domain->cycles.Total[level];}
                                                                                     printf("%12.6f\n",SecondsPerCycle*(double)total);
  printf("\n");
  printf("Total time in MGBuild   %12.6f\n",SecondsPerCycle*(double)domain->cycles.build);
  printf("Total time in MGSolve   %12.6f\n",SecondsPerCycle*(double)domain->cycles.MGSolve);
  printf("             v-cycles   %12.6f\n",SecondsPerCycle*(double)domain->cycles.vcycles);
  printf("\n\n");fflush(stdout);
}


//------------------------------------------------------------------------------------------------------------------------------
void MGBuild(domain_type * domain){
//, int e0_id, int R0_id, int homogeneous, double a, double b, double h0){
  int box,level,numLevels = domain->numLevels;

  if(domain->rank==0){printf("MGBuild...                      ");fflush(stdout);}

  // initialize timers...
  for(level=0;level<10;level++){
  domain->cycles.smooth[level]        = 0;
  domain->cycles.residual[level]      = 0;
  domain->cycles.communication[level] = 0;
  domain->cycles.restriction[level]   = 0;
  domain->cycles.interpolation[level] = 0;
  domain->cycles.norm[level]          = 0;
  domain->cycles.blas1[level]         = 0;
  domain->cycles.s2buf[level]         = 0;
  domain->cycles.buf2g[level]         = 0;
  domain->cycles.pack[level]          = 0;
  domain->cycles.unpack[level]        = 0;
  domain->cycles.bufcopy[level]       = 0;
  domain->cycles.recv[level]          = 0;
  domain->cycles.send[level]          = 0;
  domain->cycles.wait[level]          = 0;
  domain->cycles.collectives[level]   = 0;
  domain->cycles.Total[level]         = 0;
  }
  domain->cycles.build                = 0;
  domain->cycles.vcycles              = 0;
  domain->cycles.MGSolve              = 0;

  uint64_t _timeStartMGBuild = CycleTime();

  // alias all red/black masks to box0's - performance tweak to avoid having a single global list of RedBlack masks as a function of box size
  for(box=1;box<domain->numsubdomains;box++){
    for(level=0;level<numLevels-1;level++){domain->subdomains[box].levels[level].RedBlack_64bMask = domain->subdomains[0].levels[level].RedBlack_64bMask;
  }}

  // form all restrictions of alpha[] for all boxes...
  for(level=0;level<numLevels-1;level++)restriction(domain,level,__alpha,level+1,__alpha);

  // form all restrictions of beta_*[] for all boxes...
  for(level=0;level<numLevels-1;level++)restriction_betas(domain,level,level+1);

  // must communicate betas before precomputing lambda (i.e. need high values of beta from next subdomain)
  for(level=0;level<numLevels;level++){exchange_boundary(domain,level,__alpha ,1,1,1);}
  for(level=0;level<numLevels;level++){exchange_boundary(domain,level,__beta_i,1,1,1);}
  for(level=0;level<numLevels;level++){exchange_boundary(domain,level,__beta_j,1,1,1);}
  for(level=0;level<numLevels;level++){exchange_boundary(domain,level,__beta_k,1,1,1);}

  uint64_t delta = (CycleTime()-_timeStartMGBuild);
  domain->cycles.build += delta;

  if(domain->rank==0){printf("done\n");fflush(stdout);}
}


//------------------------------------------------------------------------------------------------------------------------------
void MGSolve(domain_type * domain, int e0_id, int R0_id, int homogeneous, double a, double b, double h0){ 
  // NOTE: you must have previously called MGBuild in order to guarantee all restrictions of alpha and beta have been formed and communicated
  int level,box;
  int numLevels  = domain->numLevels;

  #ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
  #endif

  if(domain->rank==0){
  if(homogeneous){printf("MGSolve...                      ");fflush(stdout);}
             else{printf("MGSolve (residual correction)...");fflush(stdout);}
  }


  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  for(level=0;level<10;level++){
  domain->cycles.smooth[level]        = 0;
  domain->cycles.residual[level]      = 0;
  domain->cycles.communication[level] = 0;
  domain->cycles.restriction[level]   = 0;
  domain->cycles.interpolation[level] = 0;
  domain->cycles.norm[level]          = 0;
  domain->cycles.blas1[level]         = 0;
  domain->cycles.s2buf[level]         = 0;
  domain->cycles.buf2g[level]         = 0;
  domain->cycles.pack[level]          = 0;
  domain->cycles.unpack[level]        = 0;
  domain->cycles.bufcopy[level]       = 0;
  domain->cycles.recv[level]          = 0;
  domain->cycles.send[level]          = 0;
  domain->cycles.wait[level]          = 0;
  domain->cycles.collectives[level]   = 0;
  domain->cycles.Total[level]         = 0;
  }
//domain->cycles.build                = 0;
  domain->cycles.vcycles              = 0;
  domain->cycles.MGSolve              = 0;


  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  uint64_t _timeStartMGSolve;
  _timeStartMGSolve = CycleTime();

  // form all restrictions of lambda[] for all boxes...
  for(level=0;level<numLevels;level++){rebuild_lambda(domain,level,a,b,h0*(double)(1<<level));}
  for(level=0;level<numLevels;level++){exchange_boundary(domain,level,__lambda,1,1,1);}


  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  if(homogeneous){
    uint64_t _timeStartVCycle = CycleTime();
    CycleMG(domain,e0_id,R0_id,a,b,h0);
    domain->cycles.vcycles += (uint64_t)(CycleTime()-_timeStartVCycle);
  }else{
    // calculate temporary RHS = residual = f-Av ...
    exchange_boundary(domain,0,e0_id,1,1,1);
    residual(domain,0,__f_minus_Av,e0_id,R0_id,a,b,h0); // rhs = f-Av
    zero_grid(domain,0,__ee);                  // ee = 0
    uint64_t _timeStartVCycle = CycleTime();
    CycleMG(domain,__ee,__f_minus_Av,a,b,h0);           // calculate the correction(ee)
    domain->cycles.vcycles += (uint64_t)(CycleTime()-_timeStartVCycle);
    add_grids(domain,0,e0_id,1.0,e0_id,1.0,__ee); // apply the correction(ee) to e0
  }

  domain->cycles.MGSolve += (uint64_t)(CycleTime()-_timeStartMGSolve);
  //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  if(domain->rank==0){printf("done\n");fflush(stdout);}
}

//------------------------------------------------------------------------------------------------------------------------------
void CycleMG(domain_type * domain, int e_id, int R_id, double a, double b, double h0){
  int level,box;
  //int SwitchToInterBoxParallelismAtLevel = 1;
  int numVCycles       = 10;
  int numSmoothsDown   = 4; // i.e. 2 Red-black relaxes
  int numSmoothsBottom = 48;
  int numSmoothsUp     = 4;
  int v,s;
  int ghosts = domain->ghosts;
  double hLevel;


  for(v=0;v<numVCycles;v++){
    // down the v-cycle...
    for(level=0;level<(domain->numLevels-1);level++){
      uint64_t _LevelStart = CycleTime();
      hLevel = h0 * (double)(1<<level);
      // relax...
      for(s=0;s<numSmoothsDown;s+=ghosts){
        if( (level>0)&&(s==0) ){         zero_grid(domain,level,e_id);}       // no need to exchange phi on first pass as it was globally initialized to zero.
        else{                    exchange_boundary(domain,level,e_id,1,1,1);} // otherwise, exchange the boundary of phi
        if((s==0)&&(ghosts>1)){
          if((level> 0)        ){exchange_boundary(domain,level,R_id,1,1,1);} // need to get RHS from neighbors if duplicating their work (need ghosts-1)
          if((level==0)&&(v==0)){exchange_boundary(domain,level,R_id,1,1,1);} // need to get RHS from neighbors if duplicating their work (need ghosts-1)
        }
        smooth(domain,level,e_id,R_id,a,b,hLevel,s);
      } // relax
      // calculate residual, restriction ...
      exchange_boundary(domain,level,e_id,1,0,0); // technically only needs to be a 1-deep ghost zone & faces only
      #ifdef __FUSION_RESIDUAL_RESTRICTION
      residual_and_restriction(domain,level,e_id,R_id,level+1,R_id,a,b,hLevel);
      #else
      residual(domain,level,__temp,e_id,R_id,a,b,hLevel);
      restriction(domain,level,__temp,level+1,R_id);
      #endif
      domain->cycles.Total[level] += (uint64_t)(CycleTime()-_LevelStart);
    } // down
 
 
  
    // bottom solve...
    uint64_t _timeBottomStart = CycleTime();
    level = domain->numLevels-1;
    hLevel = h0 * (double)(1<<level);
    #ifdef __USE_BICGSTAB
      #warning Using BiCGStab bottom solver with fixed number of iterations...
      // based on scanned page sent to me by Erin/Nick
      // Algorithm 7.7 in Iterative Methods for Sparse Linear Systems(Yousef Saad)???
      int jMax=10;
      int j=0;
      if(level>0)zero_grid(domain,level,e_id);				// e_id[] = 0
      exchange_boundary(domain,level,e_id,1,0,0);			// exchange_boundary(e_id)
      residual(domain,level,__r0,e_id,R_id,a,b,hLevel);			// r0[] = R_id[] - A(e_id)
      scale_grid(domain,level,__r,1.0,__r0);				// r[] = r0[]
      scale_grid(domain,level,__p,1.0,__r0);				// p[] = r0[]
      double r_dot_r0 = dot(domain,level,__r,__r0);			// r_dot_r0 = dot(r,r0)
      do{								// do{
        exchange_boundary(domain,level,__p,1,0,0);			//   exchange_boundary(p)
        apply_op(domain,level,__Ap,__p,a,b,hLevel);			//   Ap = A(p)
        double Ap_dot_r0 = dot(domain,level,__Ap,__r0);			//   Ap_dot_r0 = dot(Ap,r0)
        // what if Ap_dot_r0 == 0 (pivot breakdown) ???
        double alpha = r_dot_r0 / Ap_dot_r0;				//   alpha = r_dot_r0 / Ap_dot_r0
        add_grids(domain,level,__s,1.0,__r,-alpha,__Ap);		//   s[] = r[] - alpha*Ap[]
        exchange_boundary(domain,level,__s,1,0,0);			//   exchange_boundary(s)
        apply_op(domain,level,__As,__s,a,b,hLevel);			//   As = A(s)
        double As_dot_s  = dot(domain,level,__As, __s);			//   As_dot_s  = dot(As, s)
        double As_dot_As = dot(domain,level,__As,__As);			//   As_dot_As = dot(As,As)
        double omega = As_dot_s / As_dot_As;				//   omega = As_dot_s / As_dot_As
        // what if As_dot_As == 0 (converged) ???
        add_grids(domain,level,__temp,alpha,__p , omega,__s   );	//   __temp =          (alpha*p[] + omega*s[])
        add_grids(domain,level,  e_id,  1.0,e_id,   1.0,__temp);	//   e_id[] = e_id[] + (alpha*p[] + omega*s[])
        add_grids(domain,level,__r   ,  1.0,__s ,-omega,__As  );	//   r[] = s[] - omega*As[]
        double r_dot_r0_new = dot(domain,level,__r,__r0);		//   r_dot_r0_new = dot(r,r0)
        // what if r_dot_r0_new == 0 (Lanczos breakdown) ???
        // what if omega == 0 (stabilization breakdown) ???
        double beta = (r_dot_r0_new/r_dot_r0) * (alpha/omega);		//   beta = (r_dot_r0_new/r_dot_r0) * (alpha/omega)
        add_grids(domain,level,__temp,1.0,__p,-omega,__Ap  );		//   __temp =         (p[]-omega*Ap[])
        add_grids(domain,level,__p   ,1.0,__r,  beta,__temp);		//   p[] = r[] + beta*(p[]-omega*Ap[])
        r_dot_r0 = r_dot_r0_new;					//   r_dot_r0 = r_dot_r0_new   (save old r_dot_r0)
        j++;
        // FIX do convergence test (norm()) on r[] ?
      }while(j<jMax);							// }while(j<jMax);
    #else // just multiple GSRB's
      #warning Defaulting to simple GSRB bottom solver with fixed number of iterations...
      for(s=0;s<numSmoothsBottom;s+=ghosts){
        if( (level>0)&&(s==0) ){        zero_grid(domain,level,e_id);}       // no need to exchange phi on first pass as it was globally initialized to zero.
        else{                   exchange_boundary(domain,level,e_id,1,1,1);} // otherwise, exchange the boundary of phi
        if((s==0)&&(ghosts>1)){ exchange_boundary(domain,level,R_id,1,1,1);} // need to get RHS from neighbors if duplicating their work (need ghosts-1)
        smooth(domain,level,e_id,R_id,a,b,hLevel,s);
      } // relax
    #endif
    domain->cycles.Total[level] += (uint64_t)(CycleTime()-_timeBottomStart);
  
  
  
    // back up the v-cycle...
    for(level=(domain->numLevels-2);level>=0;level--){
      uint64_t _LevelStart = CycleTime();
      hLevel = h0 * (double)(1<<level);
      interpolation(domain,level+1,e_id,level,e_id);
      // relax...
      for(s=0;s<numSmoothsUp;s+=ghosts){ 
        exchange_boundary(domain,level,e_id,1,1,1);
        smooth(domain,level,e_id,R_id,a,b,hLevel,s);
      } // relax
      domain->cycles.Total[level] += (uint64_t)(CycleTime()-_LevelStart);
    } // up


    // now calculate the norm of the residual...
    #ifdef __PRINT_NORM
      uint64_t _timeStart = CycleTime();
      level = 0;
      hLevel = h0 * (double)(1<<level);
      double norm_of_residual =  0.0;
      exchange_boundary(domain,level,e_id,1,0,0);
      residual(domain,level,__temp,e_id,R_id,a,b,hLevel);
      norm_of_residual = norm(domain,level,__temp);
      uint64_t _timeNorm = CycleTime();
      domain->cycles.norm[0] += (uint64_t)(_timeNorm-_timeStart);
      if(domain->rank==0){
        if(v==0)printf("\n");
                printf("v-cycle=%2d, norm=%0.20f\n",v+1,norm_of_residual);
      }
    #endif


  } // numVCycles
}
//------------------------------------------------------------------------------------------------------------------------------
