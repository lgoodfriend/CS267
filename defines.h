//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
// Helmholtz ~ Laplacian() = a*alpha*Identity - b*Divergence*beta*Gradient
// GSRB = phi_red = phi_red + lambda(Laplacian(phi_black) - RHS)
#define  __u           0 // = what we're eventually solving for (u), cell centered
#define  __f           1 // = original right-hand side (Au=f), cell centered
#define  __alpha       2 // cell centered constant
#define  __beta_i      3 // face constant (n.b. element 0 is the left face of the ghost zone element)
#define  __beta_j      4 // face constant (n.b. element 0 is the back face of the ghost zone element)
#define  __beta_k      5 // face constant (n.b. element 0 is the bottom face of the ghost zone element)
#define  __lambda      6 // cell centered constant
#define  __ee          7 // = used for correction (ee) in residual correction form, cell centered
#define  __f_minus_Av  8 // = used for initial right-hand side (f-Av) in residual correction form, cell centered
#define  __temp        9 // = used for unrestricted residual (r), cell centered

// For BiCGStab bottom solver
#define  __r0         10
#define  __r          11
#define  __p          12
#define  __s          13
#define  __Ap         14
#define  __As         15
// For CACG
#define __Mp1         16
#define __Mp2         17
#define __Mp3         18
#define __Mr1         19
#define __Mr2         20
#define __e_id_old    21
#define ss             2 // number of s steps to do


//------------------------------------------------------------------------------------------------------------------------------
// box[j].ghost[i] = box[box[j].neighbor[i]].surface[26-i]
//------------------------------------------------------------------------------------------------------------------------------
