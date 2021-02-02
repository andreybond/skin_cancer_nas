from pathlib import Path

import numpy as np
import os

# Preprocessor configurations
#ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728/checkyourskin'),
#                Path('/mnt/data/interim/_melanoma_20200728/loc'),
#                Path('/mnt/data/interim/_melanoma_20200728/loc_old_colored')]

ROOT_PATHS = [  Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE/checkyourskin'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE/loc'),
                Path('/mnt/data/interim/_melanoma_20200728_REGISTERED_OCV_WHITE/loc_old_colored')]

# Path to data source. This folder should contain "checkyourskin", "loc", etc. folders
# MELANOMA_CLASSES = ['c43'] #, 'c44', 'c44imi'] # List of folder names associated with Melanoma (positive class)


class ClassesDefinition():
    
    def __init__(self, root_folders_list, diagnoses_names_list, class_name, int_label):
        super()
        self.root_folders_list = root_folders_list  # names of ROOT_PATHS subfolders to collect data from
        self.diagnoses_names_list = diagnoses_names_list
        self.class_name = class_name  # How our 'combined' class will be named
        self.int_label = int_label
    
    def get_samples_count(self):
        l = 0
        for root in self.root_folders_list:
            root_dir_list = list(os.listdir(str(root)))
            root_dir_list.sort()
            for diagnose_subfolder in root_dir_list:
                # now here we have to select only matching diagnoses
                if diagnose_subfolder.lower() in self.diagnoses_names_list:
                    diagnoze_subfolder_fullpath = os.path.join(str(root), diagnose_subfolder)
                    l = l + len(list(os.listdir(diagnoze_subfolder_fullpath)))
        return l
    
    def get_samples_paths(self):
        overall_list = []
        for root in self.root_folders_list:  # root folders contain subfolders with diagnoses - they are full paths
            root_dir_list = list(os.listdir(str(root)))
            root_dir_list.sort()
            for diagnose_subfolder in root_dir_list:
                # now here we have to select only matching diagnoses
                if diagnose_subfolder.lower() in self.diagnoses_names_list:
                    # we have to collect all samples from this diagnose subfolder
                    diagnose_subfolder_path = os.path.join(root, diagnose_subfolder)
                    samples_list = list(os.listdir(diagnose_subfolder_path))
                    samples_list.sort()
                    for sample in samples_list:
                        sample_full_path = os.path.join(diagnose_subfolder_path, sample)
                        overall_list.append(sample_full_path)
        return overall_list

    def get_int_label(self):
        return self.int_label

    def __str__(self):
        return 'root_folders_list=[{}], diagnoses_names_list=[{}], class_name=[{}], int_label=[{}]'.format(self.root_folders_list, self.diagnoses_names_list, self.class_name, self.int_label)



# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='C43', int_label=1), 
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='D22', int_label=0)]

# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='D22', int_label=0),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43'], class_name='C43', int_label=1),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c44'], class_name='C44', int_label=2),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l82'], class_name='L82', int_label=3) 
#                 ]

# #  5 big classes
# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03.9', 'd48'], class_name='Melanoma_like_lesions', int_label=0),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l72','d17','l72.3','d21.9','l94.2','b07.9','d23'], class_name='Non_pigmented_lesions__underskin_lesions', int_label=1),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l85.8','l85.1','l85.5','l21','l57.0','d86.3','q80','d09','l57','l85','l82'], class_name='Keratin_lesions', int_label=2),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l81.2','d81','q82.5','l98.8','l81.4','l81','l57.9','b07','l98.9','d22'], class_name='Pigmented_lesions', int_label=3), 
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['m79.81','d18','l94.0','l93','l92','l30','l40'], class_name='Blood_non_pigmented_lesions', int_label=4),]

# 3 classes C43 / D22 / Other very different class
# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=1),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['m79.81','d18','l94.0','l93','l92','l30','l40'], class_name='Blood_non_pigmented_lesions', int_label=4)]

# From 4th sept - 1st set - 3 classes 
# 1)    Melanoma: C43 + D03
# 4)    Keratin lesion: L85.8 + L85.1 + L85.5 + L21 + L57.0 + D86.3 + Q80 + D09 + L57 + L85 + L82
# 5)    Pigmented benign: L81.2 + D81 + Q82.5 + L98.8 + L81.4 + L81 + L57.9 + B07 + L98.9 + D22
# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l85.8','l85.1','l85.5','l21','l57.0','d86.3','q80','d09','l57','l85','l82'], class_name='Class2', int_label=1),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l81.2','d81','q82.5','l98.8','l81.4','l81','l57.9','b07','l98.9','d22'], class_name='Class3', int_label=2)]

# From 4th sept - 2nd set - 2 classes 
# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03.9', 'd03'], class_name='Melanoma_like_lesions', int_label=0),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l81.2','d81','q82.5','l98.8','l81.4','l81','l57.9','b07','l98.9','d22'], class_name='Class2', int_label=1)]

# # From 7th sept - 2nd set - 2 classes II variants
CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0),
                ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22', 'd81', 'l81.2','l81.4', 'q82.5'], class_name='Class2', int_label=1)]

# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43', 'd03', 'd03.9'], class_name='Melanoma_like_lesions', int_label=0),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22', 'd81', 'l81.2','l81.4', 'q82.5'], class_name='Pigmented_benign', int_label=1),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d86.3', 'l21', 'l57', 'l57.0', 'l82', 'l85', 'l85.1', 'l85.5', 'l85.8', 'q80'], class_name='Keratin_lesions', int_label=2)]

# 
# CLASSES_SET = [ ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['c43',], class_name='Melanoma', int_label=0),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['d22'], class_name='Nevuss', int_label=1),
#                 ClassesDefinition(root_folders_list=ROOT_PATHS, diagnoses_names_list=['l82'], class_name='S.Keratosis', int_label=2)]


# Generator configurations
#VALID_CHANNELS = ['r', 'ir', 'g', 'uv'] # Ordered list of required channels
VALID_CHANNELS = ['r-r', 'ir-r', 'g-g', 'uv-r', 'uv-g', 'white-g', 'white-r'] # Ordered list of required channels

IMG_WIDTH = 128
IMG_HEIGHT = 128

VALUE_MISSING = 0
MISSING_CH_IMG = np.ones((IMG_HEIGHT, IMG_WIDTH)) * VALUE_MISSING