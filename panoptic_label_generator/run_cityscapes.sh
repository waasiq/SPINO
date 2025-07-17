eval "$(conda shell.bash hook)"
conda activate spino
mkdir -p panoptic_label_generator/logs

# Original implementations (commented out)
#python panoptic_label_generator/semantic_fine_tuning.py fit --trainer.devices [0] --config panoptic_label_generator/configs/semantic_cityscapes.yaml > panoptic_label_generator/logs/adapter-logs/semantic_cityscapes.txt 2>&1
python panoptic_label_generator/boundary_fine_tuning.py fit --trainer.devices [0] --config panoptic_label_generator/configs/boundary_cityscapes.yaml > panoptic_label_generator/logs/boundary_cityscapes.txt 2>&1
#python panoptic_label_generator/instance_clustering.py test --trainer.devices [0] --config panoptic_label_generator/configs/instance_cityscapes.yaml > panoptic_label_generator/logs/adapter-logs/instance_cityscapes.txt 2>&1

#ln -s /work/dlclarge2/biswass-spino/cityscapes/camera/ /work/dlclarge2/biswass-spino/SPINO/results/cityscapes
#ln -s /work/dlclarge2/biswass-spino/cityscapes/leftImg8bit_sequence/ /work/dlclarge2/biswass-spino/SPINO/results/cityscapes

#python -m panoptic_segmentation_model.scripts_dev.evaluate_labels --dataset_name cityscapes --gpu_id 0 /work/dlclarge2/biswass-spino/cityscapes /work/dlclarge2/biswass-spino/SPINO/results/cityscapes
