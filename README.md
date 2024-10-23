# datadebias

In the metadata directory we already provide the files needed to construct the imbalanced clique sets.

For each dataset, we have:

1. clique_dict_final_coco.pkl: The cliques for COCO-GB 
2. clique_dict_final.pkl: The cliques for Waterbirds
3. clique_dict_final_urbancars.pkl: The cliques for Urbancars

To create the imbalanced clique set, we can simply run:

python src/concept_sampler.py --clique_file_name

The output of this file is a .json file containing the concept combinations to be up-sampled. 

The metadata directory contains these files as well:

1. concepts_generation.json: Concepts to be sampled for Waterbirds
2. concepts_generation_coco.json: Concepts to be sampled for COCO-GB
3. concepts_generation_urbancars.json: Concepts to be sampled for UrbanCars
   
To create the co-occurrences, we use src/co_occurence.py and src/co_occurrence_cliques.py. 

The co-occurrence code for Waterbirds, COCO-GB, and UrbanCars are a bit different due to 
the different nature of the metadata, i.e. we need different ways to extract the concepts stored
in the annotations. Therefore, we also share the different csv files for each dataset, i.e. 
co_occurrence_matrix_coco.csv, co_occurrence_matrix_urbancars.csv, co_occurrence_matrix_waterbirds.csv.
For these particular datasets, the csv files and the code are not required.

To train ConBias, run

python src/train.py --dataset <datname> --augmentation --method conbias --checkpoint_path <ckpt_path>

To evaluate CB, run

python src/evaluate.py --checkpoint_path <ckpt_path>

To evaluate OOD, run

python src/evaluate.py --checkpoint_path <ckpt_path> --type ood

NOTE: <ckpt-path> is the checkpoints saved for the base resnet model. 
NOTE: src/dataloaders.py needs to be modified with the actual dataset path on machine. 

