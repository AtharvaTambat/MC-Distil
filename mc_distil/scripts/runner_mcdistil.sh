GPU='0'


dataset="tiny-imagenet-200"
data_path="./tiny-imagenet-200/"
#dataset='cifar100'
#data_path='./cifar/'
save_dir='./meta_results_m/'

batch_size=400
eval_batch_size=400

epochs=200
lr=0.05
wd=1e-4 # metascorer paper used weight decay of 1e-4
momentum=0.95
k=5
# 1159M MACs, 80.46%
#base_name='ResNet34' 

# 555M MACs, 76.56%
#base_name='ResNet18' 

# 253M MACs, 75.25%
#base_name='ResNet10' 

# 64M MACs, 71.99%
#base_name='ResNet10_l'

# 0.8M MACs, 28.21%
#base_name='ResNet10_xxxs'

# 2M MACs, 32.05%
#base_name='ResNet10_xxs'

# 2.86M MACs, 42.99%
#base_name="ResNet10_xs"

# 4M MACs, 52.16%
base_name="ResNet10_s"
teacher_name='ResNet10'

# 16M MACs, 65.24%
#base_name='ResNet10_m'


echo "S=$base_name"
# CUDA_VISIBLE_DEVICES="$GPU"  python train_meta.py --dataset $dataset --data_path $data_path \
#     --teacher $teacher_name --model_name $base_name --save_dir $save_dir \
# 	--eval_batch_size $eval_batch_size --batch_size $batch_size  \
# 	--lr $lr --wd $wd --momentum $momentum --epochs $epochs --meta_lr 0.01 \
# 	--pretrained_student True --temperature 2 --sched_cycles 1 \
# 	--rand_seed $1
CUDA_VISIBLE_DEVICES="2"  python3 -m mc_distil.training.image_classification.train_mcdistil --dataset $dataset --data_path $data_path \
    --teacher $teacher_name --model_name $base_name --save_dir $save_dir \
	--eval_batch_size $eval_batch_size --batch_size $batch_size  \
	--lr $lr --wd $wd --momentum $momentum --epochs  200\
	--temperature 2 --rand_seed 1 
