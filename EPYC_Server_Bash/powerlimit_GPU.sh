printf "Limit GPU Max Power to 180W..."
sudo nvidia-smi -pl 180
printf "--------Print GPU Power Info--------"
nvidia-smi -q -d POWER
