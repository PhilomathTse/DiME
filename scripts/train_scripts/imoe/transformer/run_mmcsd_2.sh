export device=2
#JB DT CQ RUS UKR TOC MOC AET_HUM ANTM_CI CI_ESRX CSV_AET  FOXA_DIS
for original_target in AET_HUM
do
for lr in 5e-5
do
for temperature_rw in 1
do
for hidden_dim_rw in 256
do
for num_layer_rw in 2
do
for interaction_loss_weight in 0.5
do
for hidden_dim in 256
do
for num_patches in 4
do
for num_heads in 4
do
for seed in 3047
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_transformer.py \
    --original_target=$original_target \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw  $hidden_dim_rw \
    --num_layer_rw  $num_layer_rw \
    --interaction_loss_weight $interaction_loss_weight \
    --data mmcsd \
    --gate None \
    --train_epochs 80\
    --modality TS \
    --fusion_sparse False \
    --batch_size 64 \
    --hidden_dim $hidden_dim \
    --num_layers_fus 2 \
    --num_layers_enc 2 \
    --num_layers_pred 1 \
    --num_patches $num_patches \
    --num_experts 3 \
    --num_routers 1 \
    --top_k 1 \
    --num_heads $num_heads \
    --dropout 0.5 \
    --lr $lr \
    --n_runs 1 \
    --seed 1 \
    --gate_loss_weight 0.01 \
    --save True \
    --use_common_ids True
done
done
done
done
done
done
done
done
done
done