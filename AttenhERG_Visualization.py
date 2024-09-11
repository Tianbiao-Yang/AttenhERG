import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8) # for reproduce

import time
import numpy as np
import gc
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
#then import my own modules
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

from rdkit.Chem import rdMolDescriptors, MolSurf
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D, MolToFile, _moltoimg
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

%matplotlib inline
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.image as mpimg
from IPython.display import SVG, display
import seaborn as sns
from cairosvg import svg2png
import xlsxwriter
from PIL import Image


"""Preprocessed data"""
def data_prep(data_type):
    task_name = 'hERG'
    tasks = ['hERG']
    raw_filename = "./data/hERG_" + data_type + ".csv"
    feature_filename = raw_filename.replace('.csv','.pickle')
    filename = raw_filename.replace('.csv','')
    prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
    smiles_tasks_df = pd.read_csv(raw_filename)
    smilesList = smiles_tasks_df.smiles.values
    # print("number of all smiles: ",len(smilesList))
    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in smilesList:
        try:        
            mol = Chem.MolFromSmiles(smiles)
            atom_num_dist.append(len(mol.GetAtoms()))
            remained_smiles.append(smiles)
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
        except:
            # print("not successfully processed smiles: ", smiles)
            pass
    # print("number of successfully processed smiles: ", len(remained_smiles))
    smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
    # print(smiles_tasks_df)
    smiles_tasks_df['cano_smiles'] =canonical_smiles_list
    # assert canonical_smiles_list[8]==Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][8]), isomericSmiles=True)

    smilesList = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())<101]
    uncovered = [smiles for smiles in canonical_smiles_list if len(Chem.MolFromSmiles(smiles).GetAtoms())>100]
    smiles_tasks_df = smiles_tasks_df[~smiles_tasks_df["cano_smiles"].isin(uncovered)]

#     if os.path.isfile(feature_filename):
#         feature_dicts = pickle.load(open(feature_filename, "rb" ))
#     else:
#         feature_dicts = save_smiles_dicts(smilesList,filename)
    feature_dicts = get_smiles_dicts(smilesList)
    remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
    
    weights = []
    for i,task in enumerate(tasks):    
        negative_df = remained_df[remained_df[task] == 0][["smiles",task]]
        positive_df = remained_df[remained_df[task] == 1][["smiles",task]]
        weights.append([(positive_df.shape[0]+negative_df.shape[0])/negative_df.shape[0],\
                        (positive_df.shape[0]+negative_df.shape[0])/positive_df.shape[0]])
    
    return remained_df, smilesList, feature_dicts, weights


def eval(model, dataset, smiles_list,feature_dicts):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, test_batch in enumerate(batch_list):
        batch_df = dataset.loc[test_batch,:]
        smiles_list = batch_df.cano_smiles.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
#             validInds = np.where((y_val=='0') | (y_val=='1'))[0]
#             print(validInds)
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
#             print(validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
#             print(y_pred_adjust)
            y_pred_adjust = F.softmax(y_pred_adjust,dim=-1).data.cpu().numpy()[:,1]
            losses_list.append(loss.cpu().detach().numpy())
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
#             print(y_val,y_pred,validInds,y_val_adjust,y_pred_adjust)   

    predict_value = [[y_val_list[i], y_pred_list[i]] for i in range(len(tasks))]
    
    return  predict_value


def min_max_norm(dataset):
    if isinstance(dataset, list):
        norm_list = list()
        min_value = min(dataset)
        max_value = max(dataset)

        for value in dataset:
            tmp = (value - min_value) / (max_value - min_value)
            norm_list.append(tmp)
    return norm_list


""" Input dataset and hyperparameterization """
dataset_name = 'User_Input'
remained_df_insilico, smilesList, feature_dicts_insilico, weights = data_prep(dataset_name)
remained_df_insilico = remained_df_insilico.reset_index(drop=True)

tasks = ['hERG']
random_seed = int(time.time())
random_seed = 188
start_time = str(time.ctime()).replace(':','-').replace('  ','_').replace(' ','_')
start = time.time()

batch_size = 10
epochs = 800
p_dropout = 0.1
fingerprint_dim = 200
radius = 3
T = 2
weight_decay = 4.5
learning_rate = 3.5
per_task_output_units_num = 2 # for classification model with 2 classes
output_units_num = len(tasks) * per_task_output_units_num
time_value = ['Time','Aug','10','2023']
best_epoch = '83'
logs_names = '_'.join([str(i) for i in [time_value[-1],time_value[1],time_value[2],batch_size,str(p_dropout).split('.')[-1],
                                        fingerprint_dim,radius,T,weight_decay,learning_rate]])


""" Model parameters """

num_atom_features = 39
num_bond_features = 10

loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight),reduction='mean') for weight in weights]
model = Fingerprint(radius, T, num_atom_features,num_bond_features,
            fingerprint_dim, output_units_num, p_dropout)
model.cuda()
# tensorboard = SummaryWriter(log_dir="runs/"+start_time+"_"+prefix_filename+"_"+str(fingerprint_dim)+"_"+str(p_dropout))

# optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

# best_model_name = 'model_'+ logs_names + '_'+str(best_epoch)+'.pt'   
best_model_name = 'model_2023_Aug_10_100_1_200_3_2_4.5_3.5_83.pt'
best_model = torch.load('./saved_models/'+ best_model_name)
predict_value = eval(best_model, remained_df_insilico, smilesList,feature_dicts_insilico)


""" Interpretability Setting """
best_model_dict = best_model.state_dict()
best_model_wts = copy.deepcopy(best_model_dict)
model.load_state_dict(best_model_wts)
(best_model.align[0].weight == model.align[0].weight).all()
# Feature visualization

model_for_viz = Fingerprint_viz(radius, T, num_atom_features, num_bond_features,
            fingerprint_dim, output_units_num, p_dropout)
model_for_viz.cuda()

model_for_viz.load_state_dict(best_model_wts)
(best_model.align[0].weight == model_for_viz.align[0].weight).all()

model_for_viz.eval()
test_MAE_list = []
test_MSE_list = []

out_feature_sorted = []
out_weight_sorted = []
mol_feature_sorted = []

dataset = remained_df_insilico
test_MAE_list = []
test_MSE_list = []
valList = np.arange(0,dataset.shape[0])
batch_list = []
for i in range(0, dataset.shape[0], batch_size):
    batch = valList[i:i+batch_size]
    batch_list.append(batch) 
    
    
import matplotlib.colors as colors
import numpy as np


def truncate_colormap(cmap, minval=0.0, maxval=1.0,n=100):
    new_camp = colors.LinearSegmentedColormap.from_list(
        'trunc({n}, {a:.2f},{b:.2f})'.format(n = cmap.name, a = minval, b = maxval),
         cmap(np.linspace(minval, maxval,n)),
    )
    return new_camp


n = -1
result_list = list()
for counter, test_batch in enumerate(batch_list):
    batch_df = dataset.loc[test_batch,:]
    smiles_list = batch_df.cano_smiles.values
    y_val = batch_df[tasks[0]].values

    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts_insilico)
    atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, \
    mol_attention_weight_viz, mol_prediction = model_for_viz(
        torch.Tensor(x_atom), torch.Tensor(x_bonds),
        torch.cuda.LongTensor(x_atom_index),
        torch.cuda.LongTensor(x_bond_index),
        torch.Tensor(x_mask))

    mol_pred = np.array(mol_prediction.data.squeeze().cpu().numpy())
    atom_feature = np.stack([atom_feature_viz[L].cpu().detach().numpy() for L in range(radius+1)])
    atom_weight = np.stack([mol_attention_weight_viz[t].cpu().detach().numpy() for t in range(T)])
    mol_feature = np.stack([mol_feature_viz[t].cpu().detach().numpy() for t in range(T)])

    mol_feature_sorted.extend([mol_feature[:,i,:] for i in range(mol_feature.shape[1])])
    
    for i, smiles in enumerate(smiles_list):
        atom_num = i
        ind_mask = x_mask[i]
        ind_atom = smiles_to_rdkit_list[smiles]
        ind_feature = atom_feature[:, i]
        ind_weight = atom_weight[:, i]
        out_feature = []
        out_weight = []
        for j, one_or_zero in enumerate(list(ind_mask)):
            if one_or_zero == 1.0:
                out_feature.append(ind_feature[:,j])
                out_weight.append(ind_weight[:,j])
        out_feature_sorted.extend([out_feature[m] for m in np.argsort(ind_atom)])
        out_weight_sorted.extend([out_weight[m] for m in np.argsort(ind_atom)])        
        
        mol = Chem.MolFromSmiles(smiles)
        
        aromatic_boolean = [int(mol.GetAtomWithIdx(i).GetIsAromatic()) for i in range(mol.GetNumAtoms())]
        # if len(aromatic_boolean)>30 and np.sum(aromatic_boolean)>3:
        weight_norm = min_max_norm([out_weight[m][0] for m in np.argsort(ind_atom)])
        n = n +1
        # the prediction score
        print('Num.:         '+str(n)+'\n'+ 
              'Smiles:       '+str(smiles) +'\n'+ 
              'True label:   '+str(predict_value[0][0][n]) +'\n'+
              'Predict label:'+str(predict_value[0][1][n]) +'\n'+
              'Struture:')
        result_list.append([str(n),str(smiles),str(predict_value[0][1][n])])
        # print(n,smiles,predict_value[0][0][n],predict_value[0][1][n])
        norm = matplotlib.colors.Normalize(vmin=0,vmax=0.6)
        cmap = cm.get_cmap('coolwarm')
        trunc_colormap = truncate_colormap(cmap, 0.0, 0.8)
#         if predict_value[0][1][n] > 0.5:
#             cmap = cm.get_cmap('coolwarm') # Reds, bwr, coolwarm
#         else:
#             cmap = trunc_colormap
        # cmap = cm.get_cmap('coolwarm') # Reds, bwr, coolwarm
        plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
        atom_colors = {}
        weight_norm = np.array(weight_norm).flatten()
        threshold = weight_norm[np.argsort(weight_norm)[6]]
        weight_norm = np.where(weight_norm < threshold, 0, weight_norm)

        for i in range(len(ind_atom)):
            atom_colors[i] = plt_colors.to_rgba(float(weight_norm[i]))
        rdDepictor.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DSVG(280,280)
        drawer.SetFontSize(10)
        drawer.drawOptions().updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
        drawer.SetLineWidth(2)
        op = drawer.drawOptions()
        drawer.SetFontSize(1.3 * drawer.FontSize())

        mol = rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer.DrawMolecule(mol,highlightAtoms=range(0,len(ind_atom)),highlightBonds=[],
            highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        svg = drawer.GetDrawingText()
        svg2 = svg.replace('svg:','')
        with open('./features/svg_data/' + str(n) + '.svg','w') as wpklf:
            wpklf.write(svg2) 
        svg2png(url='./features/svg_data/'+str(n)+ '.svg', write_to='./features/svg_data/'+ str(n)+ '.png',dpi=300)
        # svg2png(url='./interpretability/svg_data/'+str(n)+ '.svg', write_to='./interpretability/svg_data/'+ str(n)+ '_t.png',dpi=600)
        svg3 = SVG(svg2)
        display(svg3)
        
        
def eval_for_viz(model, viz_list):
    model.eval()
    test_MAE_list = []
    test_MSE_list = []
    mol_prediction_list = []
    atom_feature_list = []
    atom_weight_list = []
    mol_feature_list = []
    mol_feature_unbounded_list = []
    batch_list = []
    for i in range(0, len(viz_list), batch_size):
        batch = viz_list[i:i+batch_size]
        batch_list.append(batch) 
    for counter, batch in enumerate(batch_list):        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(viz_list,get_smiles_dicts(batch))
        atom_feature_viz, atom_attention_weight_viz, mol_feature_viz, mol_feature_unbounded_viz, mol_attention_weight_viz, mol_prediction = model(
            torch.Tensor(x_atom), torch.Tensor(x_bonds),
            torch.cuda.LongTensor(x_atom_index),
            torch.cuda.LongTensor(x_bond_index),
            torch.Tensor(x_mask))

        mol_prediction_list.append(mol_prediction.cpu().detach().squeeze().numpy())
        atom_feature_list.append(np.stack([atom_feature_viz[L].cpu().detach().numpy() for L in range(radius+1)]))
        atom_weight_list.append(np.stack([mol_attention_weight_viz[t].cpu().detach().numpy() for t in range(T)]))
        mol_feature_list.append(np.stack([mol_feature_viz[t].cpu().detach().numpy() for t in range(T)]))
        mol_feature_unbounded_list.append(np.stack([mol_feature_unbounded_viz[t].cpu().detach().numpy() for t in range(T)]))
        
    mol_prediction_array = np.concatenate(mol_prediction_list,axis=0)
    atom_feature_array = np.concatenate(atom_feature_list,axis=1)
    atom_weight_array = np.concatenate(atom_weight_list,axis=1)
    mol_feature_array = np.concatenate(mol_feature_list,axis=1)
    mol_feature_unbounded_array = np.concatenate(mol_feature_unbounded_list,axis=1)
#     print(mol_prediction_array.shape, atom_feature_array.shape, atom_weight_array.shape, mol_feature_array.shape)
    return mol_prediction_array, atom_feature_array, atom_weight_array, mol_feature_array, mol_feature_unbounded_array


viz_list = ['Cc1nn(C)c(C)c1N(C(F)F)S(=O)(=O)c1c(Cl)cc(CCCC2CCN(C)CC2)cc1Cl']
    
mol_prediction_array, atom_feature_array, atom_weight_array, mol_feature_array, mol_feature_unbounded_array =  eval_for_viz(model_for_viz, viz_list)
x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(viz_list,get_smiles_dicts(viz_list))


feature_sorted = []
weight_sorted = []

for i, smiles in enumerate(viz_list):
#     draw molecules in svg format
    atom_mask = x_mask[i]
    index_atom = smiles_to_rdkit_list[smiles]
    atom_feature = atom_feature_array[:, i]
    atom_weight = atom_weight_array[:, i]
    mol_prediction = mol_prediction_array[i]
    mol_feature = mol_feature_array[:, i]
    feature_list = []
    weight_list = []
    feature_reorder = []
    weight_reorder = []
    for j, one_or_zero in enumerate(atom_mask):
        if one_or_zero == 1.0:
            feature_list.append(atom_feature[:,j])
            weight_list.append(atom_weight[:,j])
            
    feature_reorder = np.stack([feature_list[m] for m in np.argsort(index_atom)])
    weight_reorder = np.stack([weight_list[m] for m in np.argsort(index_atom)])
#     reorder for draw
    if i == 0:    
        draw_index =  [1,2,3,4,5,6,7,8,9,10,11,12,13,0,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        feature_reorder = np.stack([feature_reorder[m] for m in np.argsort(draw_index)])
        weight_reorder = np.stack([weight_list[m] for m in np.argsort(draw_index)])
    elif i == 1:
        draw_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,0,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        feature_reorder = np.stack([feature_reorder[m] for m in np.argsort(draw_index)])
        weight_reorder = np.stack([weight_list[m] for m in np.argsort(draw_index)])
    else: # using rdkit index directly
        draw_index = list(range(len(index_atom)))
#     print(feature_reorder[0].shape,weight_list[0].shape)
    feature_sorted.append(feature_reorder)
    weight_sorted.append(weight_reorder)
    
    mol = Chem.MolFromSmiles(smiles)

    drawer = rdMolDraw2D.MolDraw2DSVG(280,280)
    drawer.SetFontSize(0.56)
    op = drawer.drawOptions()
    for index, re_index in enumerate(draw_index):
        op.atomLabels[index]=mol.GetAtomWithIdx(index).GetSymbol() + str(re_index)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg2 = svg.replace('svg:','')
    svg3 = SVG(svg2)
    display(svg3)
    
    intra_mol_correlation = [np.corrcoef(feature_reorder[:,L]) for L in range(radius+1)]
    sns.set(font_scale=2)
    
    for L in range(radius+1):
        plt.figure(dpi=300)
        fig, ax = plt.subplots(figsize=(20,16))
        mask = np.zeros_like(intra_mol_correlation[L])
        mask[np.triu_indices_from(mask)] = False
        sns.heatmap(np.around(intra_mol_correlation[L],1),cmap="YlGnBu", annot=False, ax=ax, mask=mask, square=True, annot_kws={"size": 16})
    plt.show()
    plt.close()
    