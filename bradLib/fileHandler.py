
import pandas as pd
from decimal import Decimal
import numpy as np
import json

# =============================================================================
# csv to latex table converter
# =============================================================================

class csv2tab():
    
    def __init__(self, file, caption = "", alignment = ""):

        "Convert a csv file to latex table"
        
        # Reads the csv file
        file = pd.read_csv(file, header=0)
        # Rounds the values such that errors are 1sf and the values match the errors num decimal places
        file = self.numFormat(file)
        
        # initialising strings
        header = ""
        dataStr = ""
        
        # Creating an alignment string if one not entered
        if alignment == "":
            for i in file.columns:
                alignment+="c"
        
        # Creating the header string
        for col in file.columns:
            header += col
            if col != file.columns[len(file.columns)-1]:
                header += " & "
        
        # Creating the data string
        for row in file.index:
            for col in file.columns:
                
                value = file.loc[row,col]
                    
                dataStr += str(value)
                
                if (col != file.columns[-1]):
                    dataStr += " & "
                    
            dataStr += r"\\" + "\n"
        
        # Information that is always included to format the table
        tableStart = r"""\begin{table}[htb]
\centering""" + r"\caption{" + caption + r"}" + r"""
\begin{tabular}{""" + alignment + r"}" + r"""
\toprule
"""
        tableEnd = r"""\bottomrule
\end{tabular}
\label{tab:}
\end{table}"""
        
        # Printing the latex code
        print(tableStart + header + "\\\\\n\\midrule\n" + dataStr + tableEnd)
    
    def sfRound(self, x, sf=1):
        """Round to given significant figures"""
        return float('%s' % float('%.1g' % x))
    
    def numFormat(self, data):
    
        # lists for storage
        newCols = []
        remCols = []
        # Pulling the column headers 
        cols = list(data.columns)
        
        # Looping through the columns and checking if they are errors
        for i in range(len(data.columns)):
            
            # Checking if the column contains errors or data
            if cols[i].find("Error") == -1:
                column = []
                # If the column is at the end of the dataframe, it is saved
                if cols[i] == cols[len(cols)-1]:
                    newCols.append(data[cols[i]])
                # Checking if the adjacent column contains error data
                elif cols[i+1].find("Error") != -1:
                    # Rounding the errors and data
                    for j in data.index:
                        data.loc[j, cols[i+1]] = self.sfRound(data.loc[j, cols[i+1]])
                    data[cols[i+1]] = data[cols[i+1]].astype(str)
                        
                    for j in data.index:
                        rnd = int(('%.0E' % Decimal(data.loc[j,cols[i+1]]))[-2:])
                        
                        data.loc[j, cols[i]] = round(data.loc[j, cols[i]], rnd)
                        
                        column.append(f"${data.loc[j, cols[i]]}\\pm{data.loc[j, cols[i+1]]}$")
                        
                    newCols.append(column)
                    remCols.append(cols[i+1])
                # If previous conditions not passed, the column is just saved
                else:
                    newCols.append(data[cols[i]])
                
            else:
                pass
        # Removing the error columns from the list of headers 
        for col in remCols:
            cols.remove(col)
            
        return pd.DataFrame(np.array(newCols).T, columns=cols)
    
class jsonHandler():
    """
    A utility class for saving and loading json files
    """
    def save(dict, filePath):
        """
        Save dictionary to a json file
        
        Parameters
        ----------
        dict : dict
            The dictionary to be saved to json.
        filePath : str
            The path to the json file to save to.
        """
        
        # Saving the dictionary to json
        with open(filePath, "w") as file:
            json.dump(dict, file, indent=4)
    
    def load(filePath):
        """
        Load dictionary from json file
        
        Parameters
        ----------
        filePath : str
            The path to the json file to load.

        Returns
        -------
        data : dict
            The dictionary loaded from the json.
        """
        
        # Loading the json file
        with open(filePath, "r") as file:
            data = json.load(file)
            
        return data