import os
import sys
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
import os
import sys
from functools import partial
import glob
# sys.path.append("/usr/local/lib/python3.10/dist-packages/lmdb-1.4.1-py3.10-linux-x86_64.egg")
# os.chdir("/content")

def smi2_2Dcoords(smi:str):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates

def smi2_3Dcoords(smi:str,cnt:int=10,seed:int=42):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list

def inner_smi2coords(content:tuple,seed:int=42):
    smi = content[0]
    target = content[1:]
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt,seed)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol':mol,'smi': smi, 'target': target}, protocol=-1)

def smi2coords(content:tuple,seed:int=42):
    try:
        return inner_smi2coords(content,seed)
    except:
        print("failed smiles: {}".format(content[0]))
        return None



def write_lmdb(file, nthreads:int=16, seed:int=42):
    print("Generate lmdb data...")
    # outpath=os.path.dirname(file)
    df = pd.read_csv(file)
    out_lmdb=file.replace('.csv','.lmdb')
    # valid = pd.read_csv(os.path.join(inpath,'valid.csv'))
    # test = pd.read_csv(os.path.join(inpath,'test.csv'))
    
    for name, content_list in [(out_lmdb, zip(*[df[c].values.tolist() for c in df])),]:
    # for name, content_list in [('test.lmdb', zip(*[test[c].values.tolist() for c in test]))]:
        env_new = lmdb.open(
            out_lmdb,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(partial(smi2coords,seed=seed), content_list)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), inner_output)
                    i += 1
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()
    


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,help=('data to process.\n'
        'it can be:\n 1)a .csv file, \n'
        'whose first col is smiles and other cols are target to learn.\n'
        '2) a directory contains one or more .csv files,\n'
        ' with same rule described above'))
    parser.add_argument('--seed',required=False,default=42,
        type=int,help=('(optional) seed for split file and generate 3d conformation.'
                       'default=42'))
    parser.add_argument('--state',required=True,default='train',
        type=str)
    
    
    args=parser.parse_args()
    data_path:str=args.data
    seed:int=args.seed
    state:str=args.state

    
    if state=='train':

        for i in glob.glob(os.path.join((data_path.replace('.csv','')),'*.csv')):
            write_lmdb(i,seed=seed)
        
        
    elif state=='test':
        write_lmdb(data_path,seed=seed)
        

        
    
        
#   write_lmdb(inpath='/content/data', outpath='/content/data', nthreads=16)