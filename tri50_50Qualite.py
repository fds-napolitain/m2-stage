#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
from os import path
import shutil
import random
import time
import csv
import time
from PIL import Image

sys.path.insert(1, os.getcwd())  # rajoute dans le path le dossier de script

sys.argv = ["rien", "metadata/metadata_portraits.csv",
            "datasets/BASE DE DONNEE PORTRAITS",
            "datasets/Qualite", 1, 20
            ]

metadata = sys.argv[1]
list_of_dictionaries = []


### lister des dictionnaires comprenant le nom de l'image et la classe "FaceQual" :

def parse_csv_metadata(metadata):
    with open(metadata) as f:
        header = next(f)
        if header is not None:
            i = 0
            for line in f.readlines():
                # print(line)
                data = line.split(',')
                if len(data) > 15:
                    row_contents = {'filename': data[0].strip(), 'FaceQual': data[15].strip()}
                    list_of_dictionaries.append(row_contents)
                i = i + 1
                # print(row_contents)
                # print(i)

            return list_of_dictionaries


dico = parse_csv_metadata(metadata)
# print(dico)


FaceQual0 = []
FaceQual1 = []
FaceQual2 = []
FaceQual3 = []

### Séparer les différentes FaceQual :

for element in dico:
    if "FaceQual0" in element['FaceQual']:
        FaceQual0.append(element['filename'])
    elif "FaceQual1" in element['FaceQual']:
        FaceQual1.append(element['filename'])
    elif "FaceQual2" in element['FaceQual']:
        FaceQual2.append(element['filename'])
    elif "FaceQual3" in element['FaceQual']:
        FaceQual3.append(element['filename'])


###Copier les images correspondant à la liste de dictionnaires depuis un dossier source, vers un dossier cible :

def copy_image_to_set(images_list, src, dirtvt, name):
    if not os.path.exists(dirtvt):
        os.makedirs(dirtvt)
    #    widgets=[' [', progressbar.Percentage(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',]
    #    bar = progressbar.ProgressBar(maxval=len(images_list),widgets=widgets).start()
    i = 0
    print('Copying {} set.....'.format(name))
    for root, dirs, files in os.walk(src):
        # print('r:',root, 'd:',dirs, 'f:',files)
        for _file in files:
            if _file in images_list:
                # If we find it, notify and copy it to NewPath
                dirfile = os.path.join(root, _file)
                print('Found file in: ', dirfile)
                shutil.copy(dirfile, dirtvt)
                i = i + 1
    #            bar.update(i)
    #    bar.finish()
    print('\n', i, 'images in', name, 'set')
    print('\n\n')


###Récupérer aléatoirement autant de dictionnaires de chaque liste que le quadruple de "FaceQual0" :
"""
FaceQual1 = random.sample(FaceQual1, len(FaceQual0) * 4)
FaceQual2 = random.sample(FaceQual2, len(FaceQual0) * 4)
FaceQual3 = random.sample(FaceQual3, len(FaceQual0) * 4)
"""

###Répartition au sein des dossiers :

dirFaceQual0 = os.path.join(sys.argv[3], 'FaceQual0')
copy_image_to_set(FaceQual0, sys.argv[2], dirFaceQual0, 'FaceQual0')

dirFaceQual1 = os.path.join(sys.argv[3], 'FaceQual1')
copy_image_to_set(FaceQual1, sys.argv[2], dirFaceQual1, 'FaceQual1')

dirFaceQual2 = os.path.join(sys.argv[3], 'FaceQual2')
copy_image_to_set(FaceQual2, sys.argv[2], dirFaceQual2, 'FaceQual2')

dirFaceQual3 = os.path.join(sys.argv[3], 'FaceQual3')
copy_image_to_set(FaceQual3, sys.argv[2], dirFaceQual3, 'FaceQual3')


def copy_image_augment(images_list, dirtvt, name):
    #    widgets=[' [', progressbar.Percentage(), '] ',progressbar.Bar(),' (', progressbar.ETA(), ') ',]
    #    bar = progressbar.ProgressBar(maxval=len(images_list),widgets=widgets).start()
    i = 0
    print('Copying {} set.....'.format(name))
    for root, dirs, files in os.walk(dirtvt):
        # print('r:',root, 'd:',dirs, 'f:',files)
        for _file in files:
            if _file in images_list:
                # If we find it, notify and copy it to NewPath
                dirfile = os.path.join(root, _file)
                print('Found file in: ', dirfile)
                namecopy = "Copie" + str(i) + ".jpg"
                namecopy = os.path.join(dirtvt, namecopy)
                shutil.copy(dirfile, namecopy)
                img = Image.open(namecopy).transpose(Image.FLIP_LEFT_RIGHT)
                img.save(namecopy, "JPEG")
                i = i + 1

                namecopy = "Copie" + str(i) + ".jpg"
                namecopy = os.path.join(dirtvt, namecopy)
                shutil.copy(dirfile, namecopy)
                i = i + 1

                namecopy = "Copie" + str(i) + ".jpg"
                namecopy = os.path.join(dirtvt, namecopy)
                shutil.copy(dirfile, namecopy)
                i = i + 1
    #            bar.update(i)
    #    bar.finish()
    print('\n', i, 'images in', name, 'set')
    print('\n\n')


# copy_image_augment(FaceQual0, dirFaceQual0, 'FaceQual0')

print("FaceQual0 : ", len(FaceQual0))
print("FaceQual1 : ", len(FaceQual1))
print("FaceQual2 : ", len(FaceQual2))
print("FaceQual3 : ", len(FaceQual3))