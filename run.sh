#!/bin/bash

python -u blse.py -e 20 -lr 0.01 --alpha 0.0 --debug
python -u blse.py -e 20 -lr 0.01 --alpha 0.00001 --debug
python -u blse.py -e 20 -lr 0.01 --alpha 0.0001 --debug
python -u blse.py -e 20 -lr 0.01 --alpha 0.001 --debug
python -u blse.py -e 20 -lr 0.01 --alpha 0.01 --debug
python -u blse.py -e 20 -lr 0.01 --alpha 0.1 --debug
python -u blse.py -e 20 -lr 0.01 --alpha 1.0 --debug
