#!/bin/bash
printf "Start mounting ...\n\n"

printf "Mounting nvme0n1 to /mnt/dev : \n"
sudo mount /dev/nvme0n1 /mnt/dev
printf "Mounting sda to /mnt/backup : \n"
sudo mount /dev/sda /mnt/backup
sudo chown -R erebus:erebus /mnt/backup

printf "Finished ...\n\n"
