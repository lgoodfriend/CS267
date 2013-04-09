//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
typedef struct {
  struct {int i, j, k;}low;                  // global coordinates of the first (non-ghost) element of subdomain
  struct {int i, j, k;}dim;                  // dimensions of this box's core (not counting ghost zone)
  struct {int i, j, k;}dim_with_ghosts;      // dimensions of this box's core (not counting ghost zone)
  int ghosts;                                // ghost zone depth
  int pencil,plane,volume;                   // useful for offsets
  int numGrids;
  int                          bufsizes[27]; // = sizes of extracted surfaces and ghost zones (pointer to array of 27 elements)
  double    * __restrict__ surface_bufs[27]; // = extracted surface (rhs on the way down, correction on the way up)
  double    * __restrict__   ghost_bufs[27]; // = incoming ghost zone (rhs on the way down, correction on the way up)
  double   ** __restrict__ grids;            // grids[g] = pointer to grid for component g
  uint64_t  * __restrict__ RedBlack_64bMask; // Red/Black Mask (i.e. 0x0000000000000000ull or 0xFFFFFFFFFFFFFFFFull) within one plane (dim_with_ghosts^2)
  uint8_t   * __restrict__ RedBlack_4bMask;  // Red/Black 4bit bit mask (i.e. 4 elements = 0000b ... 1111b ) for the whole volume.  Ideally, you can convert a 4b bitmask into a 256b bitmask
} box_type;
//------------------------------------------------------------------------------------------------------------------------------
void destroy_box(box_type *box);
 int create_box(box_type *box, int numGrids, int low_i, int low_j, int low_k, int dim_i, int dim_j, int dim_k, int ghosts);
