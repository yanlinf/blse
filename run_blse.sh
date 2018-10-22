for a in 0.0001 0.001 0.01 0.1
do
    for lang in es ca eu
    do
        python blse.py --en_$lang --pickle -bi --save_path checkpoints/en-$lang-blse-a$a-bi.bin
        python blse.py --en_$lang --pickle --save_path checkpoints/en-$lang-blse-a$a.bin
    done
done
