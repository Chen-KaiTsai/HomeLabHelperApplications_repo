sudo powertop --auto-tune
sudo cpupower -c all frequency-set -g powersave
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
