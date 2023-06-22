"""
This file contains all necessary process and pre-process functions,
using these functions allows to work faster with protein sequences
"""


## to read fasta files
from Bio import SeqIO

## basic importations
import os
import numpy as np
import pandas as pd

## pytorch modules
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence 

## a data process module
from sklearn.model_selection import train_test_split
from collections import Counter

# for typing our functions
from typing import Tuple, Dict, List, Callable, Any

##  1_ preprocess functions

def read_file(path : str) -> pd.DataFrame:
    
    """function to read a fasta file and return a pandas.DataFrame with ['Description', 'Sequence'] as columns
    """
    
    sequences = []
    for record in SeqIO.parse(path, 'fasta'):
        sequence = str(record.seq)
        description = str(record.description)
        sequences.append(tuple([description, sequence]))
    return pd.DataFrame(sequences, columns=["Description", "Sequence"])



def read_files_to_one(dir_path : str):
    
    """reads all fasta files in a directory and returns a pd.DataFrame
    """
    
    files_n = os.listdir(dir_path)
    all_df = pd.concat([read_file(dir_path+'/'+f) for f in files_n], join='inner', ignore_index=True)
    return all_df


def concat_data(data_list : List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict] : # if you don't want to wrute pandas function manually
    
    """concat some dataframes to one. all dataframes must conains named colums 'Class' and 'Sequence'
    
    Returns a tuple with data and a dict with all class names
    """
    data = pd.concat([df[["Class", "Sequence"]] for df in data_list], join='inner', ignore_index=True)
    
    # clean data
    data.dropna(subset = "Sequence", inplace=True)
    data.drop_duplicates(subset="Sequence", inplace=True)
    data = data[~(data["Sequence"].str.contains("X") | data["Sequence"].str.contains("B") | data["Sequence"].str.contains("O") | data["Sequence"].str.contains("U") | data["Sequence"].str.contains("Z"))]
    
    return data, {c : i for i, c in enumerate(list(data["Class"].unique()))}


## 2_ process functions

def build_alhabet(corpus : pd.DataFrame) -> List[str]:
    
    """build alphabet 

    Returns:
        _type_: _description_
    """
    
    counter = Counter()
    for seq in corpus["Sequence"]:
        counter.update(seq)
    return sorted(list(counter))


def build_bin_vocab(alphabet : list) -> Dict[str, int]:
    
    """function to build label encoder vocabulary 
    """
    
    return {acid : i+1 for i, acid in enumerate(alphabet)}


def build_kmer_vocab(alphabet : list,
                     k : int) -> Dict[str, int]:
    
    """function to build k-mers vocabulary
    """
    
    import itertools
    return {kmer: i+1 for i, kmer in enumerate([''.join(x) for x in itertools.product(alphabet, repeat=k)])}



def encoding(mode : str ,
             vocab : Dict[str, int], 
             k : int = 3
             ) -> Callable:
    """Returns an encoding function between label_encoding and kmer_encoding

    Args:
        mode (str): label_encoder or kmer_encoder
        vocab (Dict[str, int]): vacbulary associated to encoding mode
        k (int, optional): lenght of k-mers if using kmer encoding

    Returns:
        Callable: encoding functions that takes sequence text in argument
    """
    
    def bin_encoding(seq : str) -> List[int]:
        
        """function to encode a textual sequence using label encoder

        Returns:
            np.array: array containing numerical transformation of a str sequence 
        """
        return [vocab[c] for c in list(seq)]
    
    def kmer_encoding(seq : str) -> List[int]:
        
        """function to encode a sequence text using kmer-encoder method

        Returns:
            np.array: array containing numerical transformation of a sequence text
        """
        kmers = []
        for i in range(len(seq) - k+1):
            kmers.append(seq[i:i+k]) 
        return [vocab[i] for i in kmers]
    
    if mode == 'label_encoder':
        return bin_encoding
    elif mode == 'kmer_encoder':
        return kmer_encoding
    else:
        print("mode must be : label_encoder or kmer_encoder")
        return 0
        


def split_data(data : pd.DataFrame,
               classes : Dict[str, int],
               train_size : float,
               seed : int) -> Tuple[List, List, List]:
    
    """split data to train, validation and test samples

    Returns:
        List: list contains tuples (label, sequence str)
    """
    data["Label"] = data["Class"].apply(lambda x : classes[x])
    data = list(zip(data["Label"].values, data["Sequence"].values))
    train, test = train_test_split(data, train_size=train_size, random_state=seed)
    valid, test = train_test_split(test, train_size=0.5, random_state=seed)
    return train, valid, test

## 2 torch special process functions

class TextDataset(Dataset):
    
    """Class to convert a data_list of tuples to torch.utils.data.Dataset type. torch.utils.data.Dataset is the correct data type to pass into DataLoaders
    and it's very easy to configure when your data is a list of tuples !
    There is a general way to manually define a torch.utils.data.Dataset object when using text data
    """
    
    def __init__(self,
                 data : List[Tuple[int, str]],              # data must be converted to tuple (label, sequence)
                 encoding_fn : Callable[[str], np.array]):  # encoding function can be bin_encoding or kmer_encoding
        
        self.data = data
        self.encoding_fn = encoding_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        label, sequence = self.data[index]
        
        # Encoding of the sequence using the encoding function passed as an argument
        encoded_sequence = self.encoding_fn(sequence)
        
        # Converting label to tensor
        label_tensor = torch.tensor(label)
        
        # Conversion of the encoded sequence into a tensor
        sequence_tensor = torch.tensor(encoded_sequence)
        
        return sequence_tensor, label_tensor


## collate function for dataloaders

def get_collate_fn() -> Callable: 
    
    def collate_fn(batch) -> Tuple[Any, torch.tensor, torch.tensor]:
        sequences, labels = zip(*batch)
        
        # Sort sequences by length in descending order, required when you want to use pack_padded with gpu
        sorted_indices = sorted(range(len(sequences)), key=lambda i: len(sequences[i]), reverse=True)
        
        sequences = [sequences[i] for i in sorted_indices] # sorted by lengths 
        labels = [labels[i] for i in sorted_indices]
        
        # Pad sequences to the same length
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.)
        
        # Store the actual length of each sequence
        sequence_lengths = [len(sequence) for sequence in sequences]  # Use a list instead of a tensor
          
        return padded_sequences, torch.tensor(labels), torch.tensor(sequence_lengths)
    return collate_fn

## 

def get_weigths_for_wsampler(data_list : List[Tuple[int, str]]) -> List[float]:
    
    """this function makes it possible to allocate weights to the indexes
    according to their label for use in a WeightedRandomSampler.

    Returns:
        list of weights for each data index in data_list
    """
    # groups all indexes by class
    class_indices = {}
    for i, (label, seq) in enumerate(data_list):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
        
    # assigns a weight based on class frequency in data passed
    class_weights = {}
    for label, indices in class_indices.items():
        class_weights[label] = 1.0 / len(indices)

    weights = [class_weights[label] for label, _ in data_list]
    
    return weights

"""
Function to get corret weights if you want to use WeightedRandomSampler 
"""
def get_weihts_for_cross_entropy(data_list : List[Tuple[int, str]]) -> torch.tensor:
    from sklearn.utils.class_weight import compute_class_weight
    """compute class weights for unbalanced datasets

    Returns:
        list of weights stored in the labels
    """
    
    class_list = [c[0] for c in data_list]
    class_weights = compute_class_weight(class_weight='balanced', classes =np.unique(class_list),y= class_list)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    return torch.FloatTensor( list( class_weights_dict.values() ) )
