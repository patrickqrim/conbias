import json
import pandas as pd
import numpy as np
import pickle
import ast

# data = pd.read_csv('/home/rwiddhi/datadebias/metadata/updated_metadata_cleaned.csv')

# landbird_df = data[(data['y'] == 0) & (data['split'] == 0)]
# waterbird_df = data[(data['y'] == 1) & (data['split'] == 0)]
# captions_landbird = landbird_df['description'].values
# captions_waterbird = waterbird_df['description'].values

# landbird_cliques = {}
# waterbird_cliques = {}

#Read gender_data
with open('/media/exx/HDD/rwiddhic/coco/annotations/gender_annotations_corrected.json', 'r') as file:
    gender_data = json.load(file)


man_cliques = {}
woman_cliques = {}

# #Find max objects in image for each class
max_objects_man = max([len(item) for item in gender_data['man'].keys()])
max_objects_woman = max([len(item) for item in gender_data['woman'].keys()])

print(max_objects_man)
print(max_objects_woman)

with open('/home/rwiddhi/datadebias/metadata/common_cliques_filtered_coco.json', 'r') as f:
    filtered_cliques = json.load(f)

print("Number of cliques: ", len(filtered_cliques))
#For each clique in filtered cliques, check if the clique is present in the captions, and update the landbird_cliques and waterbird_cliques
c = 1
for clique in filtered_cliques:
    print("Processing clique: ", c)
    clique = tuple(clique)
    man_cliques[clique] = 0
    woman_cliques[clique] = 0
    for item in gender_data['man'].keys():
        img_clique = gender_data['man'][item]
        img_clique = [word.lower() for word in img_clique if word.lower() != 'person']
        if all([word.lower() in clique for word in img_clique]):
            man_cliques[clique] += 1

    for item in gender_data['woman'].keys():
        img_clique = gender_data['woman'][item]
        if all([word.lower() in clique for word in img_clique]):
            woman_cliques[clique] += 1

    c += 1
print(man_cliques)
print(woman_cliques)

"""
We save the cliques as a dict in the following format

{
  k: {
    combo_tuple: {
      class_name: n
     }
  }
}

"""

clique_dict = {}
for k in range(2, 5):
    clique_dict[k] = {}
    for clique in filtered_cliques:
        if len(clique) == k:
            combo_tuple = tuple(clique)
            clique_dict[k][combo_tuple] = {
                'man': man_cliques[combo_tuple],
                'woman': woman_cliques[combo_tuple]
            }

print(clique_dict)

#Save as pkl
with open('/home/rwiddhi/datadebias/metadata/clique_dict_final_coco.pkl', 'wb') as f:
    pickle.dump(clique_dict, f)


# with open('/home/rwiddhi/datadebias/metadata/common_cliques_filtered_urbancars.json', 'r') as f:
#     filtered_cliques = json.load(f)

# print("Number of cliques: ", len(filtered_cliques))

# urban_cliques = {}
# country_cliques = {}

# obj_annotations_data = pd.read_csv('/home/rwiddhi/datadebias/object_annotations_urbancars.csv')

# urban_data = obj_annotations_data[obj_annotations_data['obj'] == 'urban']['objects'].values
# country_data = obj_annotations_data[obj_annotations_data['obj'] == 'country']['objects'].values
# c = 1
# for clique in filtered_cliques:
#     print("Processing clique: ", c)
#     clique = tuple(clique)
#     urban_cliques[clique] = 0
#     country_cliques[clique] = 0
#     for co_occur in urban_data:
#         co_occur = ast.literal_eval(co_occur)
#         co_occur = [word.lower() for word in co_occur if word.lower() != 'urban']
#         if all([word.lower() in clique for word in co_occur]):
#             urban_cliques[clique] += 1

#     for co_occur in country_data:
#         co_occur = ast.literal_eval(co_occur)
#         co_occur = [word.lower() for word in co_occur if word.lower() != 'country']
#         if all([word.lower() in clique for word in co_occur]):
#             country_cliques[clique] += 1

#     c += 1

# #print(urban_cliques)
# #print(country_cliques)

# clique_dict = {}
# for k in range(2, 3):
#     clique_dict[k] = {}
#     for clique in filtered_cliques:
#         if len(clique) == k:
#             combo_tuple = tuple(clique)
#             clique_dict[k][combo_tuple] = {
#                 'urban': urban_cliques[combo_tuple],
#                 'country': country_cliques[combo_tuple]
#             }

# print(clique_dict)

# #Save as pkl
# with open('/home/rwiddhi/datadebias/metadata/clique_dict_final_urbancars.pkl', 'wb') as f:
#     pickle.dump(clique_dict, f)