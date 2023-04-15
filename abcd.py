import pickle
# add path
import sys
import torch
import mwptoolkit.model.Graph2Tree.graph2tree
import mwptoolkit.
sys.path.append(r"C:\Users\goelm\OneDrive\Desktop\mawps")

path = r"C:\Users\goelm\OneDrive\Desktop\mawps"

dataloader = pickle.load(open(path + '\mawps-dataloader.pkl', 'rb'))
# model = pickle.load(open(path + '\mawps-model.pkl', 'rb'))
dataset = pickle.load(open(path + '\mawps-dataset.pkl', 'rb'))
trainer = pickle.load(open(path + '\mawps-model-trainer.pkl', 'rb'))

data = [
            {'sQuestion': 'Sita had 8 friends for dinner and they all have 8 slices each. How many slices they had in total?'
                , 'lEquations': ['37*8=x'],
             'iIndex': '122', 'lSolutions': ['4'], 'template': ''}
            # {'sQuestion': "Martin strolled to Lawrence's house. It is 12 miles from Martin's house to Lawrence's house. It took Martin 6 hours to get there. How fast did Martin go?"
            #     , 'lEquations': ['37*8=x'],
            #  'iIndex': '122', 'lSolutions': ['4'], 'template': ''}
        ]
res = dataset.preprocesss(data, use_gpu=True)
res_new = dataloader.build_batch_for_predict(res)
trainer.model.predict(res_new)