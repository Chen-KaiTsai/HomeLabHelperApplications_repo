printf "Start Applying Power Saving Settings...\n\n"

printf "\n\nStart powertop\n\n"
sudo powertop --auto-tune

printf "\n\nStart cpupower\n\n"
sudo cpupower -c all frequency-set -g powersave

printf "\n\nPrevent CPU from overclocking (CPU will run at base freq)\n"

printf "Frequency boost:"
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost

printf "\n\nFinish\n\n"
