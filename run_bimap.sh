for i in 1
do
    for lang in es ca eu
    do
        python bimap.py --orthogonal --en_$lang  --dropout_init 0.1 --dropout_interval 10 --dropout_step 0.1 -e 100 --pickle -u --valiadation_step 1 --save_path checkpoints/en-$lang-bimap-$i.bin --debug
    done
done
