import numpy as np
import pandas as pd

PATHS = ["./data_classification/descriptive_statistics_segmented.csv","./data_classification/fft_extracted_features.csv"]

class Loader:

    def __init__(self,path:str):
        self.df_raw = pd.read_csv(path,index_col=0)
        self.name = path.split("/")[-1].split(".")[0]
    
    def separate_by_label(self,labels=list[str],sep:str='label',save_to_csv:bool=True):
        dataframes = []
        if len(labels) < 1:
            return
        if len(labels) >= 2:
            for l in labels:
                df = self.df_raw[self.df_raw[sep] == l]
                df.to_csv(f"./data_classification/{self.name}_{l}.csv") if save_to_csv else dataframes.append(df)
            

if __name__ == "__main__":
    for p in PATHS:
        load = Loader(p)
        load.separate_by_label(['MI','NORM'])