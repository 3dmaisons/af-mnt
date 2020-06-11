
# qsub -M qd212@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,gpuclass=*,osrel=*,hostname=air209 ./run-check-tf.sh
# qsub -M qd212@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,gpuclass=*,osrel=*,hostname=air208 ./train-check-af.sh

# qsub -M qd212@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,gpuclass=*,osrel=*,hostname=air209 ./run-check-afdynamic.sh

qsub -M qd212@cam.ac.uk -m bea -S /bin/bash -l queue_priority=cuda-low,tests=0,mem_grab=0M,gpuclass=*,osrel=*,hostname=air209 ./run-aaf.sh

