printf "Start Applying Power Saving Settings...\n\n"

printf "\n\nStart powertop\n\n"
sudo powertop --auto-tune

printf "\n\nStart cpupower\n\n"
sudo cpupower -c all frequency-set -g powersave

printf "\n\nPrevent CPU from overclocking (CPU will run at base freq)\n"

printf "No turbo:"
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

printf "\n\nFinish\n\n"