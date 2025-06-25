eval "$(conda shell.bash hook)"
conda activate spino
mkdir -p logs
python panoptic_label_generator/semantic_fine_tuning.py fit --trainer.devices [0] --config configs/semantic_cityscapes.yaml > logs/semantic_cityscapes.txt 2>&1
python panoptic_label_generator/boundary_fine_tuning.py fit --trainer.devices [0] --config configs/boundary_cityscapes.yaml > logs/boundary_cityscapes.txt 2>&1
python panoptic_label_generator/instance_clustering.py test --trainer.devices [0] --config configs/instance_cityscapes.yaml > logs/instance_cityscapes.txt 2>&1

# ln -s /home/data/cityscapes/camera/ /home/data/cityscapes_pseudolabels/
# ln -s /home/data/cityscapes/leftImg8bit/ /home/data/cityscapes_pseudolabels/
# ln -s /home/data/cityscapes/leftImg8bit_sequence/ /home/data/pastel/cityscapes_pseudolabels/
# python -m panoptic_segmentation_model.scripts_dev.evaluate_labels --dataset_name cityscapes --gpu_id 0 /home/data/cityscapes /home/data/cityscapes_pseudolabels/
