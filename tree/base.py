"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class TreeNode():
    def __init__(self, feature=None, threshold=None, info_gain=None, value = None):

      self.feature = feature
      self.threshold = threshold
      self.info_gain = info_gain
      self.children = {}
      self.value = value
      self.split_pt= None # Applicable only for real inputs
      

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.input_type = 'D' # Discrete is considerd default for both input and output
        self.output_type = 'D'

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # """
        # Function to train and construct the decision tree
        # """

        # # If you wish your code can have cases for different types of input and output data (discrete, real)
        # # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
        # # You may(according to your implemetation) need to call functions recursively to construct the tree. 

        
        # Check input and output types
        if check_ifreal(X.iloc[:, 0]):
            self.input_type = 'R'

        if check_ifreal(y):
            self.output_type = 'R'

        # Build the tree
        self.root = self.build_tree(X, y)
    
    def build_tree(self, X: pd.DataFrame, y: pd.Series, curr_root= None, curr_depth=0, threshold = None):
        """
        Function to train and construct the decision tree
        """

        
        if(len(y.unique()) == 1):
            return TreeNode(value = y.iloc[0]) #  Some parameters need to be added

        if curr_depth <= self.max_depth:
            best_feature, gain = opt_split_attribute(X, y, self.criterion, X.columns)
            
            if best_feature is not None:
                new_node = TreeNode(feature = best_feature)
                
                
                if self.input_type == "R":
                    val = X[best_feature].mean()
                    X_left, y_left, X_right, y_right = split_data(X, y, best_feature, val)
                    if len(y_left) != 0 and len(y_right) != 0:
                        new_node.split_pt = val
                        new_node.children[f"<={val}"] = self.build_tree(X_left.drop(columns = best_feature), y_left, new_node, curr_depth+1)
                        new_node.children[f">{val}"] = self.build_tree(X_right.drop(columns = best_feature), y_right, new_node, curr_depth+1)
                    
                else:
                    for val in X[best_feature]:
                        X_left, y_left, X_right, y_right = split_data(X, y, best_feature, val)
                        new_node.children[val] = self.build_tree(X_left.drop(columns = best_feature), y_left, new_node, curr_depth+1)

                                
                if self.output_type == "D":
                    new_node.value = y.mode().get(0, y.iloc[0])
                else:
                    new_node.value = y.mean()
                return new_node

        if self.output_type == "D":
            return TreeNode(value = y.mode().get(0, y.iloc[0]))
        else:
            return TreeNode(value = y.mean())
  

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """

        # Traverse the tree you constructed to return the predicted values for the given test inputs.
        print(type(X))   
        predictions = []
        for index, row in X.iterrows():
            node = self.root
            while node.children:
                feature_value = row[node.feature]
                if self.input_type == "D":
                    try:
                        node = node.children[feature_value]
                    except KeyError:
                        break
                else:
                    if feature_value <= node.split_pt:
                        node = node.children[f"<={node.split_pt}"]
                    elif feature_value > node.split_pt:
                        node = node.children[f">{node.split_pt}"]

            predictions.append(node.value)
        
        return pd.Series(predictions)
    

    def plot(self, node=None, indent="  ") -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if node is None:
            node = self.root

        self.print_tree_node(node, indent)

    def print_tree_node(self, node, indent):
        
        if node.children == {}:
            print(node.value)    
            return
        
        node_label = self.get_node_label(node)  # Get the expression for the current node
        
        # Print the expression to traverse to this node
        print(f"{indent}{node_label}")

        # Recursively print children based on output type
        if self.input_type == 'D':
            for child_value, child_node in node.children.items():
                print(indent + f"= {child_value}: ", end = "")
                self.print_tree_node(child_node, indent + "  ")
        else:
            print(indent + "Y:  ", end = "")
            child_nodes = list(node.children.values())
            self.print_tree_node(child_nodes[0], indent + "  ")

            print(indent + "N:  ", end = "")
            child_nodes = list(node.children.values())
            self.print_tree_node(child_nodes[1], indent + "  ")
            

    def get_node_label(self, node):
        if node.feature is None:
            return f"Class: {node.value}"
        elif self.input_type == 'R':
            return f"?(Ft: {node.feature}<={node.split_pt})"
        else:
            return f"?(Ft: {node.feature})"