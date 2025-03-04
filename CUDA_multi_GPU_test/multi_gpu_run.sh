#!/bin/bash

# Shell output color configuration
RED="\033[31m"
GREEN="\033[32m"
BLUE="\033[34m"
RESET="\033[0m"

usage="Usage: $0 executable input1 input2 input3 ... inputN\n"
detail="This script will execute the same program on different GPUs with different input (Usually images or files)\n\n"
argCount=$#

gpuCount=$(nvidia-smi -L | wc -l)

function programInit()
{
    if [ $argCount -eq "0" ]; then
        printf "${RED}${usage}${RESET}"
        printf "$detail"
        printf "==================================================\n"
        printf "There are ${GREEN}${gpuCount}${RESET} GPU can be used.\n"
        printf "Listing GPUs...\n"
        nvidia-smi -L
        printf "==================================================\n"
        exit 1
    fi
}

programInit

gpuIdx=0
executable="$1"
shift

numInputs="$#"
printf "Start distribute ${GREEN}${executable}${RESET} with ${BLUE}${numInputs}${RESET} inputs to GPUs\n\n"

while (($#)); do
    input=$1
    printf "Run ${GREEN}${input}${RESET} on ${BLUE}${gpuIdx}${RESET} GPU\n"
    export CUDA_VISIBLE_DEVICES=${gpuIdx}

    ${executable} ${input} &

    gpuIdx=$(($gpuIdx+1))

    if [ $gpuIdx -eq $gpuCount ]; then
        printf "${GREEN}Wait for jobs complete\n${RESET}"
        wait
        printf "${GREEN}Done...\nContinue...\n${RESET}"

        gpuIdx="0"
    fi

    shift
done

wait

printf "${GREEN}Finish all works\n${RESET}"
unset CUDA_VISIBLE_DEVICES
