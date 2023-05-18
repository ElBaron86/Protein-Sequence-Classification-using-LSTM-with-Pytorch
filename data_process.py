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
    
    """function to read a fasta file and return a pandas.DataFrame contains colums ['Description', 'Sequence']
    """
    
    sequences = []
    for record in SeqIO.parse(path, 'fasta'):
        sequence = str(record.seq)
        description = str(record.description)
        sequences.append(tuple([description, sequence]))
    return pd.DataFrame(sequences, columns=["Description", "Sequence"])



def read_files_to_one(dir_path : str):
    
    """read all fasta files on a directory and return a pd.DataFrame
    """
    
    files_n = os.listdir(dir_path)
    all_df = pd.concat([read_file(dir_path+'/'+f) for f in files_n], join='inner', ignore_index=True)
    return all_df


def concat_data(data_list : List[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict] :
    
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
    """Return an encoding function between label_encoding or kmer_encoding

    Args:
        mode (str): label_encoder or kmer_encoder
        vocab (Dict[str, int]): vacbulary associated to encoding mode
        k (int, optional): lenght of k-mers if using kmer encoding

    Returns:
        Callable: encoding functions that takes sequence text in argument
    """
    
    def bin_encoding(seq : str) -> List[int]:
        
        """function to encode a sequence text using label encoder method

        Returns:
            np.array: array containing numerical transformation of a sequence text
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
    
    """split data to a train, validation and test 

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
    
    """Class to convert a data list of tuples to torch.utils.data.Dataset type. torch.utils.data.Dataset is the correct data type to pass into DataLoaders
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

def get_collate_fn() -> Callable: #batch : Tuple[List[torch.tensor], List[torch.tensor]]
    
    def collate_fn(batch) -> Tuple[Any, torch.tensor]:
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



"""
Function to get corret weights if you want to use WeightedRandomSampler 
"""

def get_weigths_for_wsampler(data_list : List[Tuple[int, str]]) -> List[float]:
    
    """function that returns weighted indices for use in a WeightedRandomSampler.
    It groups the indices belonging to each class and assigns a weighting according
    to the inverse of the frequency of each class in the training data set passed as an argument.

    Returns:
        list of weights for each data index in data_list
    """
    # groups all indexes by class
    class_indices = {}
    for i, (label, seq) in enumerate(data_list):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(i)
        
    # assigns a weight based on class freaquency in data passed
    class_weights = {}
    for label, indices in class_indices.items():
        class_weights[label] = 1.0 / len(indices)

    weights = [class_weights[label] for label, _ in data_list]
    
    return weights
