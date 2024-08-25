
# """
# You can add your own functions here according to your decision tree implementation.
# There is no restriction on following the below template, these fucntions are here to simply help you.
# """


import pandas as pd
import numpy as np
from scipy.special import xlogy
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    # Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse=False)

    # Fit and transform the data
    encoded_array = encoder.fit_transform(X)
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(X.columns))
    return encoded_df


# def variance(y):
#     '''
#     Function to calculate variance, avoiding nan.
#     y: variable to calculate variance. Should be a Pandas Series.
#     '''
#     if len(y) == 1:
#         return 0
#     else:
#         return y.var()      


def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    
    """
    
    """
    try:
        return any(y % 1 != 0)  # True if any value has a non-zero decimal part
    except TypeError:
        return False 

    # print(f"Series data type: {y.dtype}")
    # print(f"Series values: {y.head()}")
    
    # try:
    #     # Check if the series contains real (non-integer) values
    #     return y.apply(lambda x: isinstance(x, (int, float)) and not (isinstance(x, int) and float(x).is_integer())).any()
    # except Exception as e:
    #     print(f"Error in check_ifreal: {e}")
    #     return False
    
    # print(f"Series Name: {y.name}")
    # print(f"Series Length: {y.size}") 
    
    # if pd.api.types.is_bool_dtype(y):
    #     return False
    # elif pd.api.types.is_any_real_numeric_dtype(y):
    #     unique_ration = y.nunique() / len(y)  # Corrected len(y)
    #     if unique_ration < 0.05:
    #         return False
    #     else:
    #         return True
    # elif pd.api.types.is_object_dtype(y):
    #     unique_ration = y.nunique() / len(y)  # Corrected len(y)
    #     if unique_ration < 0.05:
    #         return False
    #     else:
    #         return True
    # else:
    #     return False

    


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    if (check_ifreal(Y)==False):
        prob = Y.unique()
        val_cnts = Y.value_counts()
        # tot_cnt = Y.value_counts().sum()
        entropy = 0.0
        for y in val_cnts:
            proportion = y/len(Y)
            # entropy+= proportion*np.log2(proportion)
            entropy+=xlogy(proportion,proportion)/np.log(2)
        entropy=-entropy
        return entropy   
    else:
        mean = Y.mean()
        mse = ((Y- mean) ** 2).mean()
        return mse
        

    




def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    if(check_ifreal(Y)==False):  # non numeric values
        pob = Y.unique()
        val_cnts = Y.value_counts()
        gi=0
     
        for y in val_cnts:
            proportion = (y/len(Y))**2
            gi = gi + proportion
        return 1-gi


    


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """

    if check_ifreal(attr) and check_ifreal(Y):
            parent_variance = np.var(Y)
            left_variance = np.var(Y[attr <= attr.mean()])
            right_variance = np.var(Y[attr > attr.mean()])
            weights = [len(Y[attr <= attr.mean()])/len(Y), len(Y[attr > attr.mean()])/len(Y)] 
            weighted_variance = weights[0] * left_variance + weights[1] * right_variance
            return parent_variance - weighted_variance

    # Case 2: Discrete Input Real Output
    elif not check_ifreal(attr) and check_ifreal(Y):
        parent_variance = np.var(Y)
        uniq_attr = np.unique(attr)
        weighted_variances = 0
        for attribute in uniq_attr:
            Y_filtered = Y[attr == attribute]
            weight = len(Y_filtered)/len(Y)
            weighted_variances += weight * (np.var(Y_filtered))
        return(parent_variance - weighted_variances)
    

    # Discrete output ==> Use Entropy or Gini Index
    # Case 3: Real Input Discrete Output
    elif check_ifreal(attr) and not check_ifreal(Y):
        parent_impurity = entropy(Y) if criterion == "information_gain" else gini_index(Y)
        threshold = attr.mean()
        values = [attr <= threshold, attr > threshold]  # Discretize the feature
        weights = [len(Y[attr <= threshold]) / len(Y), len(Y[attr > threshold]) / len(Y)]
        weighted_impurities = 0
        for i in range(2):
            child_impurity = entropy(Y[values[i]]) if criterion == "information_gain" else gini_index(Y[values[i]])
            weighted_impurities += weights[i] * child_impurity
        
        return parent_impurity - weighted_impurities

    # Case 4: Discrete Input Discrete Output
    else:
        parent_impurity = entropy(Y) if criterion == "information_gain" else gini_index(Y)
        uniq_attr = np.unique(attr)
        weighted_impurities = 0
        for attribute in uniq_attr:
            Y_filtered = Y[attr == attribute]
            weight = len(Y_filtered)/len(Y)
            child_impurity = entropy(Y_filtered) if criterion == "information_gain" else gini_index(Y_filtered)
            weighted_impurities += weight * child_impurity

        return parent_impurity - weighted_impurities




def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

    max_gain = -float('inf') 
    best_feature = None

    for feature in features:
        attr = X[feature]
        gain = information_gain(y, attr, criterion)  
        if gain > max_gain:
            max_gain = gain
            best_feature = feature

    return best_feature, max_gain




def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    if check_ifreal(X[attribute]): # Real Input
        mask = X[attribute] <= value
    else: # Discrete Input
        mask = X[attribute] == value    

    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]

    return X_left, y_left, X_right, y_right


