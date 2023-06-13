#!/bin/bash

function run() {

  jobfile=$2
  cuda=$1

  export CUDA_VISIBLE_DEVICES=$cuda
  export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
  export OMPI_MCA_opal_cuda_support=true
  while true; do
    command=$(head -n 1 $jobfile)
    if [ -z "$command" ]; then
      break
    fi
    echo $command
    sed -i '1d' $jobfile
    eval $command
  done
}

jobfile=$1

h=${HOSTNAME:0:5}

if [[ $h == "al-81" || \
$h == "al-82" || \
$h == "al-83" || \
$h == "al-84" || \
$h == "al-85" || \
$h == "al-86" || \
$h == "al-87" || \
$h == "al-88" || \
$h == "al-89" || \
$h == "bb-ma" ]]; then
  run 0 $jobfile &
#  sleep 1
#  run 1 $jobfile &
#  sleep 1
#  run 2 $jobfile &
#  sleep 1
#  run 3 $jobfile &
#  sleep 1
  run 4 $jobfile &
  sleep 1
  run 5 $jobfile &
  sleep 1
  run 6 $jobfile &
  sleep 1
  run 7 $jobfile &
  sleep 1
fi