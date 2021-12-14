#!/bin/bash

T=`date +%m%d%H%M`
# ROOT=../../

show_help() {
cat << EOF
Usage:
    ${0##*/} [-h/--help] [-p/--partition] [-n/--num] [-g/--gpu] [-c/--config] [-e/--env-name]
EOF
}

EODROOT=/mnt/lustre/qiaolei/code/EOD/eod
PLUGINROOT=/mnt/lustre/qiaolei/code/TRACKING/EOD-Tracking/tracking
config=config.yaml
partition=UCG_Share
GPU=8
NUM=2
ENV_NAME=s0.3.4

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit
            ;;
        -p|--partition)
            partition=$2
            shift 2
            ;;
        -n|--num)
            NUM=$2
            shift 2
            ;;
        -g|--gpu)
            GPU=$2
            shift 2
            ;;
        -c|--config)
            config=$2
            shift 2
            ;;
        -e|--env-name)
            ENV_NAME=$2
            shift 2
            ;;
        *)
            echo invalid arg [$1]
            show_help
            exit 1
            ;;
    esac
done

# source $ENV_NAME
which python

eod_env_setup() {
    export PLUGINPATH=$PLUGINROOT
    echo "PLUGINPATH: $PLUGINROOT"

    # export EODROOT=$EODROOT
    export PYTHONPATH=$EODROOT:$PYTHONPATH
    echo "EODROOT: $EODROOT"
}

eod_env_setup

TOTAL=`expr $GPU \* $NUM`
cat<<EOF
training ${config} on $partition, use $NUM node(s), $GPU gpu(s), total $TOTAL card(s).
EOF


spring.submit run -p $partition -n$TOTAL \
    --job-name=$(basename `pwd`) \
    --gpu \
    --cpus-per-task 5 \
"python -m eod train \
  --config=$config \
  --display=1 \
  2>&1 | tee log.train.$T.$(basename $config) "
