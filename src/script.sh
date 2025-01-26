####################################
#For Wirefix model on all sequences data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence news_series db_split yahoo_split --lr 5e-4 --bsize 32 --epoch 5 --seed $seed  --method wirefix --freeze_pretrain >> logs/Wirefix_all_${seed}.txt
done
####################################
#For Wireneigh model on all sequences data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence news_series db_split yahoo_split --lr 5e-4 --bsize 32 --epoch 5 --seed $seed  --method wireneigh --freeze_pretrain >> logs/Wireneigh_all_${seed}.txt
done
####################################
#For clora model on all sequences data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence news_series db_split yahoo_split --lr 5e-4 --bsize 32 --epoch 5 --seed $seed  --method clora --freeze_pretrain >> logs/Wireneigh_all_${seed}.txt
done
####################################
#For ER model on news_series data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence news_series --lr 3e-5 --bsize 32 --epoch 3 --seed $seed  --method origin --er >> logs/Finetune_ER_news_${seed}.txt
done

####################################
#For A-GEM model on news_series data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence news_series --lr 3e-5 --bsize 32 --epoch 3 --seed $seed  --method origin --agem >> logs/Finetune_AGEM_news_${seed}.txt
done

####################################
#For ERACE model on news_series data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence news_series --lr 3e-5 --bsize 32 --epoch 3 --seed $seed  --method origin --erace >> logs/Finetune_ERACE_news_${seed}.txt
done

####################################
#For adapter model on Yahoo data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence yahoo_split --lr 5e-5 --bsize 32 --epoch 20 --seed $seed  --method adapter --freeze_pretrain >> logs/Adapter_yahoo_${seed}.txt
done

####################################
#For prefix-tuning model on Yahoo data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence yahoo_split --lr 1e-3 --bsize 32 --epoch 20 --seed $seed  --method pv2 --freeze_pretrain >> logs/Prefix_yahoo_${seed}.txt
done

####################################
#For l2p model on Yahoo data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence news_series --lr 1e-3 --bsize 32 --epoch 20 --seed $seed  --method l2p --freeze_pretrain >> logs/L2P_yahoo_${seed}.txt
done

####################################
#For coda model on db data
####################################
for seed in 508 39 123 4536 67
do 
    CUDA_VISIBLE_DEVICES=0 python main.py --sequence db_split --lr 1e-3 --bsize 32 --epoch 10 --seed $seed  --method coda --freeze_pretrain >> logs/CODA_db_${seed}.txt
done