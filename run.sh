#!/bin/bash

python -u blse.py -e 10 -lr 0.01
python -u blse.py -e 10 -lr 0.01 --alpha 0.0
python -u blse.py -e 10 -lr 0.03 --alpha 0.0
python -u blse.py -e 10 -lr 0.1 --alpha 0.0
python -u blse.py -e 10 -lr 0.01 --alpha 1.0
python -u blse.py -e 10 -lr 0.03 --alpha 1.0
python -u blse.py -e 10 -lr 0.1 --alpha 1.0