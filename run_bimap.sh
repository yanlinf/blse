for i in `seq 5`
do
    for lang in es ca eu
    do
        python bimap.py --en_$lang --pickle -u --save_path checkpoints/en-$lang-ubi-$i.bin
    done
done
