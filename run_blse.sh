for a in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 0.9
do
    for lang in es ca eu
    do
        python blse.py --en_$lang -lr 0.01 -a $a --pickle -bi --save_path checkpoints/en-$lang-blse-a$a-bi.bin
        python blse.py --en_$lang -lr 0.01 -a $a --pickle --save_path checkpoints/en-$lang-blse-a$a.bin
    done
done
python cnn_eval.py checkpoints/*blse* -o log/blse-cnn.csv
