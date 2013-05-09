//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#ifdef _MPI
#include <mpi.h>
#endif
//------------------------------------------------------------------------------------------------------------------------------
typedef struct {
  int rank;							// MPI rank of remote process
  int local_index;						// index in subdomains[] on remote process
  #ifdef _MPI
  struct{int buf;struct{int faces,edges,corners;}offset;}send;	// i.e. calculate offset as faceSize*faces + edgeSize*edges + cornerSize*corners
  struct{int buf;struct{int faces,edges,corners;}offset;}recv;	// i.e. calculate offset as faceSize*faces + edgeSize*edges + cornerSize*corners
  #endif
} neighbor_type;

//------------------------------------------------------------------------------------------------------------------------------
typedef struct {
  struct {int i, j, k;}low;  			// global coordinates of the first (non-ghost) element of subdomain at the finest resolution
  struct {int i, j, k;}dim;  			// subdomain dimensions at finest resolution
  int numLevels;      				// number of levels in MG v-cycle.  1=no restrictions
  int ghosts;                			// ghost zone depth
  neighbor_type neighbors[27];			// MPI rank and local index (on remote process) of each subdomain neighboring this subdomain
  box_type * levels;				// pointer to an array of all coarsenings of this box
} subdomain_type;


//------------------------------------------------------------------------------------------------------------------------------

typedef struct {
  // timing information...
  struct {
    uint64_t        smooth[10];
    uint64_t      residual[10];
    uint64_t   restriction[10];
    uint64_t interpolation[10];
    uint64_t communication[10];
    uint64_t         s2buf[10];
    uint64_t          pack[10];
    uint64_t       bufcopy[10];
    uint64_t        unpack[10];
    uint64_t         buf2g[10];
    uint64_t          recv[10];
    uint64_t          send[10];
    uint64_t          wait[10];
    uint64_t          norm[10];
    uint64_t         blas1[10];
    uint64_t   collectives[10];
    uint64_t         Total[10];
    uint64_t build;   // total time spent building the coefficients...
    uint64_t vcycles; // total time spent in all vcycles (all CycleMG)
    uint64_t MGSolve; // total time spent in MGSolve
  }cycles;

  int                                   rank_of_neighbor[27]; // = MPI rank of the neighbors of this process's subdomains (presumes rectahedral packing)
  #ifdef _MPI
//MPI_Request                               send_request[27]; // to be used for non-blocking isend's
//MPI_Request                               recv_request[27]; // to be used for non-blocking irecv's
  double * __restrict__                      send_buffer[27]; // = MPI send buffers (one per neighbor)
  double * __restrict__                      recv_buffer[27]; // = MPI recieve buffer (one per neighbor)
  struct{int faces,edges,corners;}           buffer_size[27]; // = MPI buffer size (one per neighbor) in the units of faces/edges/corners
  #endif

// n.b. i=unit stride
  struct {int i, j, k;}dim;			// global dimensions at finest resolution
  struct {int i, j, k;}subdomains_in;		// total number of subdomains in i,j,k
  struct {int i, j, k;}subdomains_per_rank_in;	// number of subdomains in i,j,k
  int rank;					// MPI rank of this process
  int numsubdomains;				// number of subdomains owned by this process
  int numLevels;				// number of levels in MG v-cycle.  1=no restrictions
  int numGrids;                                 // number of grids (variables)
  int ghosts;					// ghost zone depth
  subdomain_type * subdomains;			// pointer to a list of all subdomains owned by this process
} domain_type;

//------------------------------------------------------------------------------------------------------------------------------
 int create_subdomain(subdomain_type * box, 
                      int subdomain_low_i, int subdomain_low_j, int subdomain_low_k,
                      int subdomain_dim_i, int subdomain_dim_j, int subdomain_dim_k,
                      int numGrids, int ghosts, int numLevels, int ss);
void destroy_domain(domain_type * domain);
 int create_domain(domain_type * domain,
                   int subdomain_dim_i, int subdomain_dim_j, int subdomain_dim_k,
                   int subdomains_per_rank_in_i, int subdomains_per_rank_in_j, int subdomains_per_rank_in_k,
                   int ranks_in_i, int ranks_in_j, int ranks_in_k,
                   int rank, int numGrids, int ghosts, int numLevels, int ss);
void MGBuild(domain_type * domain);
void MGSolve(domain_type * domain, int e0_id, int R0_id, int homogeneous, double a, double b, double h0, int ss);
void CycleMG(domain_type * domain, int e_id, int R_id, double a, double b, double h0, int ss);
void print_timing(domain_type *domain);
//------------------------------------------------------------------------------------------------------------------------------
