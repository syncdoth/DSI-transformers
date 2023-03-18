#!/bin/bash
source /mnt/home/schoiaj/.Miniconda3/etc/profile.d/conda.sh
conda activate dsi

bash run.sh ${TACC_WORKDIR} 0,1,2,3