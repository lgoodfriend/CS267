//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdint.h>
#include "../timer.h"
//------------------------------------------------------------------------------------------------------------------------------
// Exchange boundaries by aggregating into domain buffers
//------------------------------------------------------------------------------------------------------------------------------
void exchange_boundary(domain_type *domain, int level, int grid_id, int exchange_faces, int exchange_edges, int exchange_corners){
  uint64_t _timeCommunicationStart = CycleTime();
  uint64_t _timeStart,_timeEnd;
  int sendBox,recvBox,n;
  //                 { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}
  int       di[27] = {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1};
  int       dj[27] = {-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1,-1,-1,-1, 0, 0, 0, 1, 1, 1};
  int       dk[27] = {-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int    faces[27] = {0,0,0,0,1,0,0,0,0,  0,1,0,1,0,1,0,1,0,  0,0,0,0,1,0,0,0,0};
  int    edges[27] = {0,1,0,1,0,1,0,1,0,  1,0,1,0,0,0,1,0,1,  0,1,0,1,0,1,0,1,0};
  int  corners[27] = {1,0,1,0,0,0,1,0,1,  0,0,0,0,0,0,0,0,0,  1,0,1,0,0,0,1,0,1};
  int exchange[27] = {0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0};
  //int neighbors_to_be_exchanged[27];int NumNeighbors=0;

  for(n=0;n<27;n++){
    if(                       exchange_faces   )exchange[n] |=   faces[n];
    if( (domain->ghosts>1) && exchange_edges   )exchange[n] |=   edges[n];
    if( (domain->ghosts>1) && exchange_corners )exchange[n] |= corners[n];
                                     //if(n!=13)exchange[n] |= 1; // apples to apples comparison (no one can ever skip edges/corners)
    //if(exchange[n])neighbors_to_be_exchanged[NumNeighbors++]=n;
  }


  #ifdef _MPI
  int   FaceSizeAtLevel = domain->subdomains[0].levels[level].dim.i*domain->subdomains[0].levels[level].dim.i*domain->ghosts;
  int   EdgeSizeAtLevel = domain->subdomains[0].levels[level].dim.i*domain->ghosts*domain->ghosts;
  int CornerSizeAtLevel = domain->ghosts*domain->ghosts*domain->ghosts;

  MPI_Request requests[54];
  MPI_Status  status[54];
  int nMessages=0;
  #endif

  // loop through bufs, prepost Irecv's
  #ifdef _MPI
  _timeStart = CycleTime();
  for(n=0;n<27;n++)if(exchange[26-n] && (domain->rank_of_neighbor[26-n] != domain->rank) ){
    int size = FaceSizeAtLevel*domain->buffer_size[26-n].faces + EdgeSizeAtLevel*domain->buffer_size[26-n].edges + CornerSizeAtLevel*domain->buffer_size[26-n].corners;
    MPI_Irecv(domain->recv_buffer[26-n],size,MPI_DOUBLE,domain->rank_of_neighbor[26-n],n,MPI_COMM_WORLD,&requests[nMessages]);
    nMessages++;
  }
  _timeEnd = CycleTime();
  domain->cycles.recv[level] += (_timeEnd-_timeStart);
  #endif


  // extract surface, pack into surface_bufs
  _timeStart = CycleTime();
  #pragma omp parallel for private(n,sendBox) collapse(2) schedule(static,1)
  for(sendBox=0;sendBox<domain->numsubdomains;sendBox++){
    //for(nn=0;nn<NumNeighbors;nn++){n=neighbors_to_be_exchanged[nn];
    for(n=0;n<27;n++)if(exchange[n]){
    int ghosts = domain->subdomains[sendBox].levels[level].ghosts;
    int pencil = domain->subdomains[sendBox].levels[level].pencil;
    int  plane = domain->subdomains[sendBox].levels[level].plane;
    int  dim_i = domain->subdomains[sendBox].levels[level].dim.i;
    int  dim_j = domain->subdomains[sendBox].levels[level].dim.j;
    int  dim_k = domain->subdomains[sendBox].levels[level].dim.k;
      int low_i,low_j,low_k;
      int buf_i,buf_j,buf_k;
      switch(di[n]){
        case -1:low_i=ghosts;buf_i=ghosts;break;
        case  0:low_i=ghosts;buf_i= dim_i;break;
        case  1:low_i= dim_i;buf_i=ghosts;break;
      };
      switch(dj[n]){
        case -1:low_j=ghosts;buf_j=ghosts;break;
        case  0:low_j=ghosts;buf_j= dim_j;break;
        case  1:low_j= dim_j;buf_j=ghosts;break;
      };
      switch(dk[n]){
        case -1:low_k=ghosts;buf_k=ghosts;break;
        case  0:low_k=ghosts;buf_k= dim_k;break;
        case  1:low_k= dim_k;buf_k=ghosts;break;
      };
      extract_from_grid(low_i,low_j,low_k,buf_i,buf_j,buf_k,pencil,plane,domain->subdomains[sendBox].levels[level].grids[grid_id],domain->subdomains[sendBox].levels[level].surface_bufs[n],1);
    }
  }
  _timeEnd = CycleTime();
  domain->cycles.s2buf[level] += (_timeEnd-_timeStart);


  // pack domain buffers
  #ifdef _MPI
  _timeStart = CycleTime();
  #pragma omp parallel for private(n,sendBox,recvBox) collapse(2) schedule(static,1)
  for(sendBox=0;sendBox<domain->numsubdomains;sendBox++){
    //for(nn=0;nn<NumNeighbors;nn++){n=neighbors_to_be_exchanged[nn];
    for(n=0;n<27;n++)if(exchange[n]){
      recvBox = domain->subdomains[sendBox].neighbors[n].local_index;
      if(domain->subdomains[sendBox].neighbors[n].rank != domain->rank){
        buffer_copy(domain->send_buffer[domain->subdomains[sendBox].neighbors[n].send.buf] +
                        FaceSizeAtLevel*domain->subdomains[sendBox].neighbors[n].send.offset.faces +
                        EdgeSizeAtLevel*domain->subdomains[sendBox].neighbors[n].send.offset.edges +
                      CornerSizeAtLevel*domain->subdomains[sendBox].neighbors[n].send.offset.corners,
                                        domain->subdomains[sendBox].levels[level].surface_bufs[n],
                                        domain->subdomains[sendBox].levels[level].bufsizes[n], 1 );
      }
  }}
  _timeEnd = CycleTime();
  domain->cycles.pack[level] += (_timeEnd-_timeStart);
  #endif

 
  // loop through bufs, post Isend's
  #ifdef _MPI
  _timeStart = CycleTime();
  for(n=0;n<27;n++)if(exchange[n] && (domain->rank_of_neighbor[n] != domain->rank) ){
    int size = FaceSizeAtLevel*domain->buffer_size[n].faces + EdgeSizeAtLevel*domain->buffer_size[n].edges + CornerSizeAtLevel*domain->buffer_size[n].corners;
    MPI_Isend(domain->send_buffer[n],size,MPI_DOUBLE,domain->rank_of_neighbor[n],n,MPI_COMM_WORLD,&requests[nMessages]);
    nMessages++;
  }
  _timeEnd = CycleTime();
  domain->cycles.send[level] += (_timeEnd-_timeStart);
  #endif


  // exchange locally... try and hide within Isend latency... 
  _timeStart = CycleTime();
  #pragma omp parallel for private(n,sendBox,recvBox) collapse(2)  schedule(static,1)
  for(recvBox=0;recvBox<domain->numsubdomains;recvBox++){
    //for(nn=0;nn<NumNeighbors;nn++){n=neighbors_to_be_exchanged[nn];
    for(n=0;n<27;n++)if(exchange[n]){
      sendBox = domain->subdomains[recvBox].neighbors[n].local_index;
      if(domain->subdomains[recvBox].neighbors[n].rank == domain->rank){
        buffer_copy(domain->subdomains[recvBox].levels[level].ghost_bufs[n],
                    domain->subdomains[sendBox].levels[level].surface_bufs[26-n],
                    domain->subdomains[sendBox].levels[level].bufsizes[26-n], 1 );
  }}}
  _timeEnd = CycleTime();
  domain->cycles.bufcopy[level] += (_timeEnd-_timeStart);


  #ifdef _MPI
  // loop through bufs, MPI_Wait on recvs
  _timeStart = CycleTime();
  MPI_Waitall(nMessages,requests,status);
  _timeEnd = CycleTime();
  domain->cycles.wait[level] += (_timeEnd-_timeStart);
  #endif


  // unpack domain buffers 
  #ifdef _MPI
  _timeStart = CycleTime();
  #pragma omp parallel for private(n,sendBox,recvBox) collapse(2)  schedule(static,1)
  for(recvBox=0;recvBox<domain->numsubdomains;recvBox++){
    //for(nn=0;nn<NumNeighbors;nn++){n=neighbors_to_be_exchanged[nn];
    for(n=0;n<27;n++)if(exchange[n]){
      sendBox = domain->subdomains[recvBox].neighbors[n].local_index;
      if(domain->subdomains[recvBox].neighbors[n].rank != domain->rank){
        buffer_copy(          domain->subdomains[recvBox].levels[level].ghost_bufs[n],
          domain->recv_buffer[domain->subdomains[recvBox].neighbors[n].recv.buf] +
              FaceSizeAtLevel*domain->subdomains[recvBox].neighbors[n].recv.offset.faces +
              EdgeSizeAtLevel*domain->subdomains[recvBox].neighbors[n].recv.offset.edges +
            CornerSizeAtLevel*domain->subdomains[recvBox].neighbors[n].recv.offset.corners,
                              domain->subdomains[recvBox].levels[level].bufsizes[n], 1 );
      }
    }
  }
  _timeEnd = CycleTime();
  domain->cycles.unpack[level] += (_timeEnd-_timeStart);
  #endif
 
 
  // unpack ghost_bufs, insert into grid
  _timeStart = CycleTime();
  #pragma omp parallel for private(n,recvBox) collapse(2)  schedule(static,1)
  for(recvBox=0;recvBox<domain->numsubdomains;recvBox++){
    //for(nn=0;nn<NumNeighbors;nn++){n=neighbors_to_be_exchanged[nn];
    for(n=0;n<27;n++)if(exchange[n]){
    int ghosts = domain->subdomains[recvBox].levels[level].ghosts;
    int pencil = domain->subdomains[recvBox].levels[level].pencil;
    int  plane = domain->subdomains[recvBox].levels[level].plane;
    int  dim_i = domain->subdomains[recvBox].levels[level].dim.i;
    int  dim_j = domain->subdomains[recvBox].levels[level].dim.j;
    int  dim_k = domain->subdomains[recvBox].levels[level].dim.k;
      int low_i,low_j,low_k;
      int buf_i,buf_j,buf_k;
      switch(di[n]){
        case -1:low_i=           0;buf_i=ghosts;break;
        case  0:low_i=      ghosts;buf_i= dim_i;break;
        case  1:low_i=ghosts+dim_i;buf_i=ghosts;break;
      };
      switch(dj[n]){
        case -1:low_j=           0;buf_j=ghosts;break;
        case  0:low_j=      ghosts;buf_j= dim_j;break;
        case  1:low_j=ghosts+dim_j;buf_j=ghosts;break;
      };
      switch(dk[n]){
        case -1:low_k=           0;buf_k=ghosts;break;
        case  0:low_k=      ghosts;buf_k= dim_k;break;
        case  1:low_k=ghosts+dim_k;buf_k=ghosts;break;
      };
      insert_into_grid(low_i,low_j,low_k,buf_i,buf_j,buf_k,pencil,plane,domain->subdomains[recvBox].levels[level].ghost_bufs[n],domain->subdomains[recvBox].levels[level].grids[grid_id],0);
    }
  }
  _timeEnd = CycleTime();
  domain->cycles.buf2g[level] += (_timeEnd-_timeStart);

  domain->cycles.communication[level] += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
