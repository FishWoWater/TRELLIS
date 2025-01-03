set -e 
set -x 
# install requirements and depdendencies 
. ./dataset_toolkits/setup.sh

# # for objaversexl-sketchfab 
# # for sketchfab, score 6.5 -> 29k objects, 6.75 15k objects 
export DATASET_NAME=ObjaverseXL
export DATASET_SOURCE=sketchfab
export OUTPUT_DIR=datasets/ObjaverseXL_sketchfab
export WORLD_SIZE=1
export SCORE_THRESH=6.75

# # for objaversexl-github, score 6.5 -> 43k objects, 6.75 -> 22k objects 
# export DATASET_NAME=ObjaverseXL
# export DATASET_SOURCE=github
# export OUTPUT_DIR=datasets/ObjaverseXL_github
# export WORLD_SIZE=1
# export SCORE_THRESH=7.25

# for ABO 
# export DATASET_NAME=ABO
# export OUTPUT_DIR=datasets/ABO 
# export WORLD_SIZE=1
# export SCORE_THRESH=5.0 

# # for 3d-future
# export DATASET_NAME=3D-FUTURE
# export OUTPUT_DIR=datasets/3D-FUTURE
# export WORLD_SIZE=1
# export SCORE_THRESH=5.0 

# # for HSSD 
# export DATASET_NAME=HSSD
# export OUTPUT_DIR=datasets/HSSD
# export WORLD_SIZE=1
# export SCORE_THRESH=5.25

# # for Toys-4K 
# export DATASET_NAME=Toys4k
# export OUTPUT_DIR=datasets/Toys4k
# export WORLD_SIZE=1


export RANK=0
export MAX_WORKERS=1
export NUM_VIEWS=36
export NUM_VIEWS_COND=24

# initially build meta data of the specified dataset
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR}

# download dataset
python dataset_toolkits/download.py ${DATASET_NAME} --filter_low_aesthetic_score ${SCORE_THRESH} --output_dir ${OUTPUT_DIR} --world_size ${WORLD_SIZE}

# render multi-view images with blender (e.g. 150 views)
python dataset_toolkits/render.py ${DATASET_NAME} --num_views ${NUM_VIEWS} --filter_low_aesthetic_score ${SCORE_THRESH}  --output_dir ${OUTPUT_DIR} --max_workers ${MAX_WORKERS}

# voxelization (based on open3d)
python dataset_toolkits/voxelize.py ${DATASET_NAME} --filter_low_aesthetic_score ${SCORE_THRESH}  --output_dir ${OUTPUT_DIR}

# extract DINOv2 features
python dataset_toolkits/extract_feature.py --filter_low_aesthetic_score ${SCORE_THRESH} --output_dir ${OUTPUT_DIR}
# update metadata
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR}

# extract SparseStructure Latents
python dataset_toolkits/encode_ss_latent.py --filter_low_aesthetic_score ${SCORE_THRESH} --output_dir ${OUTPUT_DIR}
# update metadata
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR}

# encode to SLATS
python dataset_toolkits/encode_latent.py --filter_low_aesthetic_score ${SCORE_THRESH} --output_dir ${OUTPUT_DIR}
# update metadata manually 
python dataset_toolkits/build_metadata.py ${DATASET_NAME} --filter_low_aesthetic_score ${SCORE_THRESH} --source ${DATASET_SOURCE} --output_dir ${OUTPUT_DIR} 

# render condition images
python dataset_toolkits/render_cond.py ${DATASET_NAME} --num_views ${NUM_VIEWS_COND} --filter_low_aesthetic_score ${SCORE_THRESH} --output_dir ${OUTPUT_DIR}
