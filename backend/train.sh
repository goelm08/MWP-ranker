#!/bin/bash

cd data/GraphConstruction
python constituency.py

cd ../../src
mkdir checkpoint_dir
mkdir checkpoint_dir/valid
mkdir output

echo -----------pretrained embedding generating-----------
python pretrained_embedding.py -pretrained_embedding="C:\Users\goelm\Downloads\glove.6B.zip"
#python pretrained_embedding.py -pretrained_embedding="/content/drive/MyDrive/research/Graph2Tree/glove.6B.zip"
echo ------------Begin training---------------------------
python graph2tree.py
echo -----------------------------------------------------
python sample_valid.py 