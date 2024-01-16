#!/usr/bin/bash

array=("worker2" "worker3" "worker4" "worker5"
       "worker6" "worker7" "worker8")

for ((i=0; i<${#array[@]}; i++)); do
  echo ${array[${i}]}
  scp $1 ${array[${i}]}:$2
done

echo "Done!"
