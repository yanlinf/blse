for lang in es ca eu
do
    python artetxe.py --en_$lang --project_source --orthogonal --save_path checkpoints/en-$lang-artetxe-s-o.bin --pickle
    python artetxe.py --en_$lang --orthogonal --save_path checkpoints/en-$lang-artetxe-t-o.bin --pickle
done