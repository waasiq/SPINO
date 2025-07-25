eval "$(conda shell.bash hook)"
conda activate spino
mkdir -p panoptic_label_generator/logs

# Original implementations (commented out)
#python panoptic_label_generator/semantic_fine_tuning.py fit --trainer.devices [0] --config panoptic_label_generator/configs/semantic_cityscapes.yaml > panoptic_label_generator/logs/vit-logs/semantic_cityscapes.txt 2>&1
python panoptic_label_generator/boundary_fine_tuning.py fit --trainer.devices [0] --config panoptic_label_generator/configs/boundary_cityscapes.yaml > panoptic_label_generator/logs/comer-logs/boundary_cityscapes.txt 2>&1
#python panoptic_label_generator/instance_clustering.py test --trainer.devices [0] --config panoptic_label_generator/configs/instance_cityscapes.yaml > panoptic_label_generator/logs/vit-logs/instance_cityscapes.txt 2>&1

#ln -s /work/dlclarge2/masoodw-spino100/dl-lab/cityscapes/leftImg8bit_sequence/ /work/dlclarge2/masoodw-spino100/dl-lab/cityscapes/results/cityscapes_classic_mlp
#ln -s /work/dlclarge2/masoodw-spino100/dl-lab/cityscapes/camera/ /work/dlclarge2/masoodw-spino100/dl-lab/cityscapes/results/cityscapes_classic_mlp

#python -m panoptic_segmentation_model.scripts_dev.evaluate_labels --dataset_name cityscapes --gpu_id 0 /work/dlclarge2/masoodw-spino100/dl-lab/cityscapes /work/dlclarge2/masoodw-spino100/dl-lab/cityscapes/results/cityscapes_classic_mlp
