#! /bin/bash

python -u 2train.py --model ./checkpoints/a000001bi-250 --alpha 0.001 -lr 0.01 -e 20 --debug --binary --save_path ./checkpoints/a000001bi2