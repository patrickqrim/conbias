import os
import json
import pandas as pd
import networkx as nx
import numpy as np
import csv
import ast
import numpy as np

#Read gender_data
# with open('/media/exx/HDD/rwiddhic/Whac-A-Mole/create_datasets/urbancars/metadata/urbancars/meta_data.json', 'r') as file:
#     urbancars = json.load(file)

# objects = ['urban', 'country', 'alley', 'crosswalk', 'downtown', 'gas station', 'garage', 'driveway', 'forest road', 'field road', 'desert road', 'fireplug', 'stop sign', 'street sign', 'parking meter', 'traffic light', 'cow', 'horse', 'sheep']

# co_obj_index = {}
# co_obj_urban = urbancars['co_occur_object'][0]
# co_obj_country = urbancars['co_occur_object'][1]

# for item in co_obj_urban['class_list']:
#     cat_id = item['id']
#     cat_name = item['name']

#     co_obj_index[cat_id] = cat_name

# for item in co_obj_country['class_list']:
#     cat_id = item['id']
#     cat_name = item['name']

#     co_obj_index[cat_id] = cat_name

# print(co_obj_index)
#  #print(objects)
# #print(all_img_categories)
# # Initialize object index mapping
# obj_index = {obj: idx for idx, obj in enumerate(objects)}

# # Initialize the co-occurrence matrix
# co_occurrence_matrix = [[0] * len(objects) for _ in objects]

# obj_annotations = pd.read_csv('/media/exx/HDD/rwiddhic/urbancars_new/obj_annotations.csv')

# #print(ast.literal_eval(obj_annotations['bg'].iloc[0])[0])

# #apply lambda function to column bg of obj_annotations
# obj_annotations['bg'] = obj_annotations['bg'].apply(lambda x: ast.literal_eval(x)[0])
# obj_annotations['co_obj'] = obj_annotations['co_obj'].apply(lambda x: int(x.strip('tensor([])')))
# obj_annotations['co_obj'] = obj_annotations['co_obj'].apply(lambda x: co_obj_index[x])

# obj_annotations['objects'] = list(zip(obj_annotations['obj'], obj_annotations['bg'], obj_annotations['co_obj']))

# print(obj_annotations.head())
# print(obj_annotations.columns)

# obj_annotations.to_csv('object_annotations_urbancars.csv', index=True)

## ---URBANCARS CO-OCCURRENCE MATRIX--- ##
# df = pd.read_csv('object_annotations_urbancars.csv')
# # Updated list of objects with correct naming conventions for items in the dataset
# original_objects = ['urban', 'country', 'alley', 'crosswalk', 'downtown', 'gas_station', 'garage-outdoor', 'driveway', 'forest_road', 'field_road', 'desert_road', 'fireplug', 'stop_sign', 'street_sign', 'parking_meter', 'traffic_light', 'cow', 'horse', 'sheep']

# # Create an empty co-occurrence matrix using the corrected list of objects
# co_occurrence_matrix = pd.DataFrame(0, index=original_objects, columns=original_objects)

# # Iterate through the DataFrame rows
# for row_objects in df['objects']:
#     # Check each object from the original list against the objects in the row
#     for i, obj1 in enumerate(original_objects):
#         if obj1 in row_objects:
#             for j in range(i + 1, len(original_objects)):  # Start from the next object to prevent double counting
#                 obj2 = original_objects[j]
#                 if obj2 in row_objects:
#                     co_occurrence_matrix.loc[obj1, obj2] += 1
#                     co_occurrence_matrix.loc[obj2, obj1] += 1  # Symmetric update


# print(co_occurrence_matrix)

# co_occurrence_matrix.to_csv('/home/rwiddhi/datadebias/metadata/co_occurrence_matrix_urbancars.csv', index=True)

## --COCO CO-OCCURRENCE MATRIX-- ##

with open('/media/exx/HDD/rwiddhic/coco/annotations/unique_categories.txt', 'r') as f:
    objects = f.readlines()
    objects = [x.strip() for x in objects]

with open('/media/exx/HDD/rwiddhic/coco/annotations/gender_annotations_corrected.json', 'r') as f:
    gender_data = json.load(f)

captions_man = [caption for _, caption in gender_data['man'].items()]
captions_man = [[item if item!='person' else 'man' for item in caption] for caption in captions_man]
#captions_man = [caption.append('man') for caption in captions_man]
captions_woman = [caption for _, caption in gender_data['woman'].items()]
captions_woman = [[item if item!='person' else 'woman' for item in caption] for caption in captions_woman]
#captions_woman = [caption.append('woman') for caption in captions_woman]

all_captions = captions_man + captions_woman


obj_index = {obj: idx for idx, obj in enumerate(objects)}

# # Create an empty co-occurrence matrix using the corrected list of objects
co_occurrence_matrix = pd.DataFrame(0, index=objects, columns=objects)

# Iterate through the DataFrame rows
for caption in all_captions:
    # Check each object from the original list against the objects in the row
    for i, obj1 in enumerate(objects):
        if obj1 in caption:
            for j in range(i + 1, len(objects)):  # Start from the next object to prevent double counting
                obj2 = objects[j]
                if obj2 in caption:
                    co_occurrence_matrix.loc[obj1, obj2] += 1
                    co_occurrence_matrix.loc[obj2, obj1] += 1  # Symmetric update


print(co_occurrence_matrix)

co_occurrence_matrix.to_csv('/home/rwiddhi/datadebias/metadata/co_occurrence_matrix_coco.csv', index=True)
#
# Fill the co-occurrence matrix
# for obj_list in objects:
#     for i in range(len(obj_list)):
#         for j in range(i + 1, len(obj_list)):
#             obj1, obj2 = obj_list[i], obj_list[j]
#             if obj1 in obj_index and obj2 in obj_index:
#                 idx1, idx2 = obj_index[obj1], obj_index[obj2]
#                 co_occurrence_matrix[idx1][idx2] += 1
#                 co_occurrence_matrix[idx2][idx1] += 1  # Because it's symmetric

# with open('/home/rwiddhi/datadebias/metadata/co_occurrence_matrix_urbancars.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     # Write the header (the list of objects)
#     writer.writerow([''] + objects)  # Add an empty string for the top-left corner of the header
#     # Write the co-occurrence data
#     for obj, row in zip(objects, co_occurrence_matrix):
#         writer.writerow([obj] + row)