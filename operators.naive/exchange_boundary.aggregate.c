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
  int    faces[27] = {0,0,0,0,1,0,0,0,0,  0,1,0,1,0,1,0,1,0,  0,0,0,0,1,0,0,0,0};
  int    edges[27] = {0,1,0,1,0,1,0,1,0,  1,0,1,0,0,0,1,0,1,  0,1,0,1,0,1,0,1,0};
  int  corners[27] = {1,0,1,0,0,0,1,0,1,  0,0,0,0,0,0,0,0,0,  1,0,1,0,0,0,1,0,1};
  int exchange[27] = {0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0};
  for(n=0;n<27;n++){
    if(                       exchange_faces   )exchange[n] |=   faces[n];
    if( (domain->ghosts>1) && exchange_edges   )exchange[n] |=   edges[n];
    if( (domain->ghosts>1) && exchange_corners )exchange[n] |= corners[n];
                                     //if(n!=13)exchange[n] |= 1; // apples to apples comparison (no one can ever skip edges/corners)
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
  #pragma omp parallel for private(sendBox)
  for(sendBox=0;sendBox<domain->numsubdomains;sendBox++){__box_grid_to_surface_bufs(&domain->subdomains[sendBox].levels[level],grid_id);}
  _timeEnd = CycleTime();
  domain->cycles.s2buf[level] += (_timeEnd-_timeStart);


  // pack domain buffers
  #ifdef _MPI
  _timeStart = CycleTime();
  #pragma omp parallel for private(n,sendBox,recvBox)
  for(sendBox=0;sendBox<domain->numsubdomains;sendBox++){
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
  #pragma omp parallel for private(n,sendBox,recvBox)
  for(recvBox=0;recvBox<domain->numsubdomains;recvBox++){
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
  #pragma omp parallel for private(n,sendBox,recvBox)
  for(recvBox=0;recvBox<domain->numsubdomains;recvBox++){
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
  #pragma omp parallel for private(recvBox)
  for(recvBox=0;recvBox<domain->numsubdomains;recvBox++){__box_ghost_bufs_to_grid(&domain->subdomains[recvBox].levels[level],grid_id);}
  _timeEnd = CycleTime();
  domain->cycles.buf2g[level] += (_timeEnd-_timeStart);

  domain->cycles.communication[level] += (uint64_t)(CycleTime()-_timeCommunicationStart);
}
