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
from scipy.stats import entropy
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True
import copy
import pandas as pd
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

# %matplotlib inline
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import matplotlib.image as mpimg
# from IPython.display import SVG, display
import seaborn as sns
from cairosvg import svg2png
import xlsxwriter
from PIL import Image
import matplotlib.colors as colors
import numpy as np


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


""" Uncertainty Prediction"""

# Entropy
def entropy_uncertainty(preds_proba) -> np.ndarray:
    return np.array(entropy(np.transpose(preds_proba)))

def multi_initial(preds_probas):
    return np.var(preds_probas, axis=0).ravel()

# MC Dropout
def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def predict_Ttimes(model, dataset, feature_dicts, T=30):

    model.eval()
    model.apply(apply_dropout)

    valList = np.arange(0, dataset.shape[0])
    batch_size = 100
    tasks = ['hERG']
    loss_function_Ttimes = [nn.CrossEntropyLoss() for i in range(len(tasks))]
    per_task_output_units_num = 2

    # predict stochastic dropout model T times
    preds_times = []
    for t in range(T):
        # if t % 10 == 0: print(f'Have predicted for {t+1}/{T} times')

        y_val_list = {}
        y_pred_list = {}
        losses_list = []
        batch_list = []

        for i in range(0, dataset.shape[0], batch_size):
            batch = valList[i:i + batch_size]
            batch_list.append(batch)
        for counter, test_batch in enumerate(batch_list):
            batch_df = dataset.iloc[test_batch, :]
            smiles_list = batch_df.cano_smiles.values

            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                         feature_dicts)
            atoms_prediction, mol_prediction = model(torch.Tensor(x_atom).to('cuda'),
                                                     torch.Tensor(x_bonds).to('cuda'),
                                                     torch.cuda.LongTensor(x_atom_index),
                                                     torch.cuda.LongTensor(x_bond_index),
                                                     torch.Tensor(x_mask).to('cuda')
                                                     )

            for i, task in enumerate(tasks):
                y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                                                         per_task_output_units_num]
                y_val = batch_df[task].values

                validInds = np.where((y_val == 0) | (y_val == 1))[0]
                if len(validInds) == 0:
                    continue
                y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
                validInds = torch.cuda.LongTensor(validInds).squeeze()
                y_pred_adjust = torch.index_select(y_pred, 0, validInds)
                loss = loss_function_Ttimes[i](
                    y_pred_adjust,
                    torch.cuda.LongTensor(y_val_adjust))
                y_pred_adjust = F.softmax(y_pred_adjust, dim=-1).data.cpu().numpy()[:, 1]
                losses_list.append(loss.cpu().detach().numpy())
                try:
                    y_val_list[i].extend(y_val_adjust)
                    y_pred_list[i].extend(y_pred_adjust)
                except:
                    y_val_list[i] = []
                    y_pred_list[i] = []
                    y_val_list[i].extend(y_val_adjust)
                    y_pred_list[i].extend(y_pred_adjust)
        preds = y_pred_list[0]
        preds_times.append([p for p in preds])

    p_hat = np.array(preds_times)
    p_hat_binary = np.array([[[1 - p, p] for p in sample] for sample in p_hat])
    return p_hat_binary

def mc_dropout(preds_probas):
    '''https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/'''
    posterior_vars = np.std(preds_probas, axis=0) ** 2
    posterior_vars_c0 = posterior_vars[:, 0]
    return posterior_vars_c0


"""Predict via the AttenhERG model"""

def eval(model, dataset, smiles_list,feature_dicts,loss_function,batch_size,tasks,per_task_output_units_num):
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
    
    # Uncertainty predict 
    ## entropy_uncertainty
    y_pred_list_uncertainty = [y_pred_list[i] for i in range(len(tasks))]
    y_preds_val_uncertainty = np.array([[1-pred, pred] for pred in y_pred_list_uncertainty[0]])
    Entropy_list = entropy_uncertainty(y_preds_val_uncertainty)
    ## MCdropout_uncertainty
    mc_pred_probas = predict_Ttimes(model,dataset,feature_dicts,T=10)
    MCdropout_list = mc_dropout(mc_pred_probas)
    
    return  predict_value,Entropy_list,MCdropout_list


def min_max_norm(dataset):
    if isinstance(dataset, list):
        norm_list = list()
        min_value = min(dataset)
        max_value = max(dataset)

        for value in dataset:
            tmp = (value - min_value) / (max_value - min_value)
            norm_list.append(tmp)
    return norm_list


def truncate_colormap(cmap, minval=0.0, maxval=1.0,n=100):
    new_camp = colors.LinearSegmentedColormap.from_list(
        'trunc({n}, {a:.2f},{b:.2f})'.format(n = cmap.name, a = minval, b = maxval),
         cmap(np.linspace(minval, maxval,n)),
    )
    return new_camp


""" The User input the data """

def prep_text(input_name):
    with open('./userup/' + input_name) as rpklf:
        try:
            input_list = []
            for i_data in rpklf:
                i = i_data.rstrip('\n').split('\t')
                input_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(i[0])))
            n = len(input_list)
        except:
            print('Convert the structure into the correct SDF or SMILES text format (.smi, .csv, .txt).')
            sys.exit()
        
    return input_list,n

def prep_sdf(input_name):
    try:
        mols = Chem.SDMolSupplier('./userup/'+ input_name)
        input_list = [Chem.MolToSmiles(mol) for mol in mols]
        n = len(input_list)
    except:
        print('Convert the structure into the correct SDF or SMILES text format (.smi, .csv, .txt).')
        sys.exit()
        
    return input_list,n


def main():
    """The files name of user input """
    
    input_name = 'user_input.csv'

    if input_name.split('.')[-1] == 'sdf':
        prep_data = prep_sdf(input_name)
    elif input_name.split('.')[-1] in ['csv','txt','smi']:
        prep_data = prep_text(input_name)
    else:
        print('Convert the structure into the correct SDF or SMILES text format (.smi, .csv, .txt).')
        sys.exit()

    with open('./data/hERG_User_Input.csv','w') as wpklf:
        wpklf.write('﻿hERG,smiles' + '\n')
        if prep_data[1] == 1:
            herg = ['1','0']
            smi = [prep_data[0][0],prep_data[0][0]]
        elif prep_data[1] == 2:
            herg = ['1','0']
            smi = [prep_data[0][0],prep_data[0][1]]
        else:
            herg = ['1','0'] + [str(label) for label in list(np.random.randint(0,2,prep_data[1]-2))]
            smi = prep_data[0]
        for s in range(0,len(smi)):
            re = [herg[s],smi[s]]
            wpklf.write(','.join(re) + '\n')


    """ Input dataset and hyperparameterization """

    dataset_name = 'User_Input'
    remained_df_insilico, smilesList, feature_dicts_insilico, weights = data_prep(dataset_name)
    remained_df_insilico = remained_df_insilico.reset_index(drop=True)

    tasks = ['hERG']
    random_seed = int(time.time())
    random_seed = 188
    start_time = str(time.ctime()).replace(':','-').replace('  ','_').replace(' ','_')
    start = time.time()

    batch_size = 100
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

    best_model_name = 'model_'+ logs_names + '_'+str(best_epoch)+'.pt'   
    best_model = torch.load('./saved_models/'+ best_model_name)
    predict_value,Entropy_list,MCdropout_list = eval(best_model, remained_df_insilico, smilesList,feature_dicts_insilico, loss_function, batch_size,tasks,per_task_output_units_num)
    print('First, the prediction was finished ...')


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


    """Interpretability analysis"""

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
    #         print('Num.:         '+str(n)+'\n'+ 
    #               'Smiles:       '+str(smiles) +'\n'+ 
    #               'True label:   '+str(predict_value[0][0][n]) +'\n'+
    #               'Predict label:'+str(predict_value[0][1][n]) +'\n'+
    #               'Struture:')
            result_list.append([str(n),smiles,predict_value[0][1][n],predict_value[0][0][n],Entropy_list[n],MCdropout_list[n]])
            # print(n,smiles,predict_value[0][0][n],predict_value[0][1][n])
            norm = matplotlib.colors.Normalize(vmin=0,vmax=0.6)
            cmap = cm.get_cmap('coolwarm')
            trunc_colormap = truncate_colormap(cmap, 0.0, 0.8)
            if predict_value[0][1][n] > 0.5:
                cmap = cm.get_cmap('coolwarm') # Reds, bwr, coolwarm
            else:
                cmap = trunc_colormap
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
            drawer.SetFontSize(1)
            drawer.drawOptions().updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
            drawer.SetLineWidth(2)
            drawer.SetFontSize(1.3 * drawer.FontSize())
            op = drawer.drawOptions()

            mol = rdMolDraw2D.PrepareMolForDrawing(mol)
            drawer.DrawMolecule(mol,highlightAtoms=range(0,len(ind_atom)),highlightBonds=[],
                highlightAtomColors=atom_colors)
            drawer.FinishDrawing()

            svg = drawer.GetDrawingText()
            svg2 = svg.replace('svg:','')
            with open('./userup/svg_data/' + str(n) + '.svg','w') as wpklf:
                wpklf.write(svg2) 
            svg2png(url='./userup/svg_data/'+str(n)+ '.svg', write_to='./userup/svg_data/'+ str(n)+ '.png',dpi=300)
            # svg2png(url='./interpretability/svg_data/'+str(n)+ '.svg', write_to='./interpretability/svg_data/'+ str(n)+ '_t.png',dpi=600)
    #         svg3 = SVG(svg2)
    #         display(svg3)
    print('Second, the interpretability analysis was finished ...')


    """Shown the colorbar"""

    # # It will take some times, so we remove the block. When try run this block, check the `_t.png` being right.
    # svg_list_path = os.listdir('./interpretability/svg_data/')
    # svg_list = svg_list = [i for i in svg_list_path if i.split('.')[-1] == 'svg']

    # for svg_id in svg_list:
    #     index_id = svg_id.split('.')[0]
    #     lena= mpimg.imread('./interpretability/svg_data/' + index_id + '.png')
    #     # plt.figure(figsize=(2.8,3.0))
    #     plt.imshow(lena)# 显示图片
    #     plt.axis('off')# 不显示坐标轴
    #     plt.colorbar(plt_colors)
    #     plt.savefig('./interpretability/svg_data/' + index_id + '_t.png', dpi=300)
    #     plt.show()


    """Save the result to the Excel files"""    

    index_List = [i[0] for i in result_list]
    pd.set_option('display.float_format',lambda x : '%.9f' % x)
    df=pd.DataFrame(result_list,columns=['Index_ID','Smiles','Predict_Score','Label','Entropy_Uncertainty','MCDropout_Uncertainty'],index=index_List) 
    if prep_data[1] == 1:
        df = df.head(1)
    else:
        df = df
    path = r"./userup/svg_data/"
    pics = os.listdir(path)
    # Define the names of Excel and worksheets to be written
    book = xlsxwriter.Workbook(r"./userup/result_" + input_name.split('.')[0] + ".xlsx")
    sheet = book.add_worksheet("hERG")

    # Define the name of the two columns, and then fill in the nicknames to match.
    cell_format = book.add_format({'bold':True, 'font_size':16,'font_name':'Times New Roman','align':'center'})
    cell_format_1 = book.add_format({'font_size':16,'font_name':'Times New Roman','align':'center','align':'vcenter'})
    cell_format_1.set_align('center')
    cell_format_1.set_align('vcenter')
    sheet.set_column('A:A', 20)
    sheet.set_column('C:C', 20)
    # sheet.set_column('D:D', 15)
    sheet.set_column('D:D', 30)
    sheet.set_column('E:E', 35)
    sheet.set_column('F:F', 15)
    sheet.write("A1", "Index_ID",cell_format)
    sheet.write("B1", "Structure",cell_format)
    sheet.write("C1", "Predict_Score",cell_format)
    # sheet.write("D1", "Label",cell_format)
    sheet.write("D1", "Entropy_Uncertainty",cell_format)
    sheet.write("E1", "MCDropout_Uncertainty",cell_format)
    sheet.write("F1", "Smiles",cell_format)
    sheet.write_column(1, 0, df.Index_ID.values.tolist(),cell_format_1)
    sheet.write_column(1, 2, df.Predict_Score.values.tolist(),cell_format_1)
    # sheet.write_column(1, 3, df.Label.values.tolist(),cell_format_1)
    sheet.write_column(1, 3, df.Entropy_Uncertainty.values.tolist(),cell_format_1)
    sheet.write_column(1, 4, df.MCDropout_Uncertainty.values.tolist(),cell_format_1)
    sheet.write_column(1, 5, df.Smiles.values.tolist(),cell_format_1)


    # To fix the size of the image, the cell where the picture is inserted must also be resized
    image_width = 280
    image_height = 280
    cell_width = 42
    cell_height = 240
    sheet.set_column("B:B", cell_width) # Set cell column width
    for i in range(len(df.Index_ID.values.tolist())):
        if df.Index_ID.values.tolist()[i] + ".png" in pics:
            # Fixed width / width of the original picture to be inserted
            x_scale = image_width / (Image.open(os.path.join(path, df.Index_ID.values.tolist()[i] + ".png")).size[0]) 
            # Fixed height / height of the original picture to be inserted
            y_scale = image_height / (Image.open(os.path.join(path, df.Index_ID.values.tolist()[i] + ".png")).size[1]) 
            sheet.set_row(i + 1, cell_height) # Set the row height
            sheet.insert_image(
                "B{}".format(i + 2),
                os.path.join(path, df.Index_ID.values.tolist()[i] + ".png"),
                {"x_scale": x_scale, "y_scale": y_scale, "x_offset": 15, "y_offset": 20},
            )  # Set the x_offset and y_offset so that the image is centered as much as possible
    sheet.set_zoom(zoom=50)
    book.close()
    # Remove the svg picture
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
    print('Finally, the result of prediction score and analysis structure were saved in ./userup/')

    
if __name__ == '__main__':
    main()
    
    