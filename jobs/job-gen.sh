#!/bin/bash

for i in {1..8} ; do
		cat <<EOF > job.hopper.cacg.$i
#PBS -N CommunicationAvoidingMultiGrid
#PBS -o results/results.hopper.cacg.$i.\$PBS_JOBID
#PBS -j oe
#PBS -q regular
#PBS -l walltime=01:00:00
#PBS -l mppwidth=$((i*i*i))
#PBS -l mppdepth=6
#PBS -l mppnppn=4

set -x
cd \$PBS_O_WORKDIR
module swap PrgEnv-pgi PrgEnv-intel

export OMP_NUM_THREAD=6

EOF
		
	for j in 1 ; do
		for k in 5 ; do
			if [ $i -eq 1 ] ; then
				echo aprun -n $((i*i*i)) -d 6 -N 1 -S 1 -ss -cc numa_node ./test.hopper.ompif.bicgstab $k $j $j $j $i $i $i >> job.hopper.cacg.$i
				echo aprun -n $((i*i*i)) -d 6 -N 1 -S 1 -ss -cc numa_node ./test.hopper.ompif.cg $k $j $j $j $i $i $i >> job.hopper.cacg.$i
				echo aprun -n $((i*i*i)) -d 6 -N 1 -S 1 -ss -cc numa_node ./test.hopper.ompif.cacg $k $j $j $j $i $i $i 2 >> job.hopper.cacg.$i
				echo aprun -n $((i*i*i)) -d 6 -N 1 -S 1 -ss -cc numa_node ./test.hopper.ompif.cacg $k $j $j $j $i $i $i 4 >> job.hopper.cacg.$i
			else 
				echo aprun -n $((i*i*i)) -d 6 -N 4 -S 1 -ss -cc numa_node ./test.hopper.ompif.bicgstab $k $j $j $j $i $i $i >> job.hopper.cacg.$i
				echo aprun -n $((i*i*i)) -d 6 -N 4 -S 1 -ss -cc numa_node ./test.hopper.ompif.cg $k $j $j $j $i $i $i >> job.hopper.cacg.$i
				echo aprun -n $((i*i*i)) -d 6 -N 4 -S 1 -ss -cc numa_node ./test.hopper.ompif.cacg $k $j $j $j $i $i $i 2 >> job.hopper.cacg.$i
				echo aprun -n $((i*i*i)) -d 6 -N 4 -S 1 -ss -cc numa_node ./test.hopper.ompif.cacg $k $j $j $j $i $i $i 4 >> job.hopper.cacg.$i
			fi
		done
	done
done
