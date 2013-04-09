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
#include "timer.h"
#include "defines.h"
#include "box.h"
#include "mg.h"
#include "operators.h"
//------------------------------------------------------------------------------------------------------------------------------
#include "operators.naive/buffer_copy.c"
#include "operators.naive/ghosts.c"
#include "operators.naive/exchange_boundary.aggregate.c"
#include "operators.naive/smooth.c"

#include "operators.naive/residual.c"
#include "operators.naive/restriction.c"
#include "operators.naive/interpolation.c"
#include "operators.naive/misc.c"
#include "operators.naive/max_norm.c"
#include "operators.naive/dot.c"
//------------------------------------------------------------------------------------------------------------------------------
