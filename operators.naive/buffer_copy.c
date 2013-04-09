//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
void buffer_copy(double * __restrict__ destination, double * __restrict__ source, int elements, int useCacheBypass){
  memcpy(destination,source,elements*sizeof(double));
}
