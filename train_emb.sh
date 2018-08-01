#!/bin/bash

# time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15

for lang in eu ca es
do
    ./word2vec/trunk/word2vec -train "plain/$lang.txt" -output "emb/$lang.bin" -cbow 0 -size 300 -window 5 -negative 15 -sample 1e-4 -binary 1 -save-vocab "vocab/$lang.txt" -threads 16
done 
