#!/bin/bash
# First Argument: model
# Second Argument: clientname
# Third Argument: hostname
# All other arguments are gpu numbers

if [ "$1" == "-h" ]; then
  echo "Arg1: model"
  echo "Arg2: clientname"
  echo "Arg3: hostname"
  echo "Next X args GPU indicies"
  exit 0
fi

tmux new-session -d -s client
tmux send-keys -t client:0 "python caffe_client.py $1 $2$4 $3 --gpu_to_use $4" C-m

for(( i=5; i<=$#; i++ )); do
    tmux new-window -t client:$(($i - 4))
    tmux send-keys -t client:$(($i - 4)) "python caffe_client.py $1 $2${!i} $3 --gpu_to_use ${!i}" C-m
done
