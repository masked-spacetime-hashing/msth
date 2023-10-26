#!/bin/bash
for i in {1..20}; do
    bash MSTH/scripts/run.sh 3 anoynmous_method_${i} wandb
    rm -r tmp
done
