#
# For Hopper
#
# Based on the compile file with the section for hopper.

# TODO: Add header necessary header files to requirements
HEADERS =
EXTRA = bench.c mg.c box.c timer.x86.c
TARGETS = run.hopper.naive run.hopper.reference run.hopper.fusion \
	  run.hopper.ompif test.hopper.ompif.cg test.hopper.ompif.bicgstab \
	  test.hopper.ompif.cacg \

all:	$(TARGETS)

run.hopper.naive: $(EXTRA) operators.naive.c
	cc -O3 -fno-alias -fno-fnalias -msse3 -openmp $(EXTRA) \
		operators.naive.c \
		-D_MPI -D__PRINT_COMMUNICATION_BREAKDOWN -o $@

test.hopper.ompif.bicgstab: $(EXTRA) operators.ompif.c
	cc -O3 -fno-alias -fno-fnalias -msse3 -openmp $(EXTRA) \
		operators.ompif.c \
		-D_MPI -D__PRINT_COMMUNICATION_BREAKDOWN -D__USE_BICGSTAB \
		-o $@ -g
test.hopper.ompif.cg: $(EXTRA) operators.ompif.c
	cc -O0 -fno-alias -fno-fnalias -check-pointers=write -msse3 -openmp $(EXTRA) \
		operators.ompif.c -traceback -check=stack,uninit \
		-D_MPI -D__PRINT_COMMUNICATION_BREAKDOWN -D__USE_CG \
		-o $@ -g

test.hopper.ompif.cacg: $(EXTRA) operators.ompif.c
	cc -O0 -fno-alias -fno-fnalias -msse3 -openmp $(EXTRA) \
		operators.ompif.c -traceback -check=stack,uninit \
		-D__PRINT_NORM -D_MPI -D__PRINT_COMMUNICATION_BREAKDOWN -D__USE_CACG \
		-o $@ -g

run.hopper.reference: $(EXTRA) operators.reference.c
	cc -O3 -fno-alias -fno-fnalias -msse3 -openmp $(EXTRA) \
		operators.reference.c \
		-D_MPI -D__PRINT_COMMUNICATION_BREAKDOWN -o $@

run.hopper.fusion: $(EXTRA) operators.reference.c
	cc -O3 -fno-alias -fno-fnalias -msse3 -openmp $(EXTRA) \
		operators.reference.c \
		-D_MPI -D__PRINT_COMMUNICATION_BREAKDOWN \
		-D__FUSION_RESIDUAL_RESTRICTION -o $@

run.hopper.ompif: $(EXTRA) operators.ompif.c
	cc -O3 -fno-alias -fno-fnalias -msse3 -openmp $(EXTRA) \
		operators.ompif.c \
		-D_MPI -D__PRINT_COMMUNICATION_BREAKDOWN \
		-D__FUSION_RESIDUAL_RESTRICTION -D__COLLABORATIVE_THREADING=6 \
		-o $@

clean:
	rm -f $(TARGETS)
