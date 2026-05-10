from scipy.stats import kurtosis, skew
import numpy as np


#NAO PRECISA INSTANCIAR

class calculate_kurtosis_skewness:

    @classmethod
    def calc_skew(cls, data):
        return skew(data,bias=False)
    
    @classmethod
    def calc_kurtosis(cls,data):
        return kurtosis(data,fisher=True,bias=False)