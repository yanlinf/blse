for a in 0.0001 0.001 0.01 0.1 0.3 0.5 0.7 0.9
do
    for lang in es ca eu
    do
        python -u blse.py --en_$lang -lr 0.01 -a $a --pickle -bi --save_path checkpoints/en-$lang-blse-a$a-bi.bin
        python -u blse.py --en_$lang -lr 0.01 -a $a --pickle --save_path checkpoints/en-$lang-blse-a$a.bin
    done
done
python -u convert.py checkpoints/en-??-blse* ubise
python -u dan_eval.py checkpoints/en-??-blse* -o log/blse-1203-dan.csv
