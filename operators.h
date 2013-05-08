//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
void                  apply_op(domain_type *domain, int level, int Ax_id, int x_id, double a, double b, double hLevel, int deep);
  void                    smooth(domain_type *domain, int level, int phi_id, int rhs_id, double a, double b, double hLevel, int sweep);
  void                  residual(domain_type *domain, int level, int res_id, int phi_id, int rhs_id, double a, double b, double hLevel);
  void             interpolation(domain_type *domain, int level_c, int id_c, int level_f, int id_f);
  void               restriction(domain_type *domain, int level_f, int id_f, int level_c, int id_c);
  void         restriction_betas(domain_type *domain, int level_f, int level_c);
  void  residual_and_restriction(domain_type *domain, int level_f, int phi_id, int rhs_id, int level_c, int res_id, double a, double b, double hLevel);

//double          norm_of_residual(domain_type *domain, int level, int phi_id, int rhs_id, double a, double b, double hLevel);
double                       dot(domain_type *domain, int level, int id_a, int id_b);
double                      norm(domain_type *domain, int level, int grid_id);
  void            rebuild_lambda(domain_type *domain, int level, double a, double b, double hLevel);
  void                 add_grids(domain_type *domain, int level, int id_c, double scale_a, int id_a, double scale_b, int id_b);
  void                scale_grid(domain_type *domain, int level, int id_c, double scale_a, int id_a);
  void                 zero_grid(domain_type *domain, int level, int grid_id);
  void initialize_grid_to_scalar(domain_type *domain, int level, int grid_id, double h, double scalar);
  void         exchange_boundary(domain_type *domain, int level, int grid_id, int exchange_faces, int exchange_edges, int exchange_corners);


  void buffer_copy(double * __restrict__ destination, double * __restrict__ source, int elements, int useCacheBypass);


//------------------------------------------------------------------------------------------------------------------------------
  void __box_smooth_GSRB_multiple(box_type *box, int phi_id, int rhs_id, double a, double b, double h, int sweep);
  void __box_smooth_GSRB_multiple_threaded(box_type *box, int phi_id, int rhs_id, double a, double b, double h, int sweep);
  void __box_residual(box_type *box, int res_id, int phi_id, int rhs_id, double a, double b, double h);
  void __box_residual_and_restriction(box_type *fine, int phi_id, int rhs_id, box_type *coarse, int res_id, double a, double b, double h);
  void __box_restriction(box_type *fine, int id_f, box_type *coarse, int id_c);
  void __box_restriction_betas(box_type *fine, box_type *coarse);
  void __box_interpolation(box_type *coarse, int id_c, box_type *fine, int id_f);
  void __box_grid_to_surface_bufs(box_type *box, int grid_id);
  void __box_ghost_bufs_to_grid(box_type *box, int grid_id);
  void __box_zero_grid(box_type *box,int grid_id);
  void __box_rebuild_lambda(box_type *box, double a, double b, double h);
  void __box_add_grids(box_type *box, int id_c, double scale_a, int id_a, double scale_b, int id_b);
  void __box_scale_grid(box_type *box, int id_c, double scale_a, int id_a);
  void __box_initialize_grid_to_scalar(box_type *box, int grid_id, double h, double scalar);
