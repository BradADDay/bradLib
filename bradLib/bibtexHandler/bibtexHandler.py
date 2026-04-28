
# A bibtex handling script, inspired by https://github.com/NayanDusoruth/NayanGeneralUtils/tree/main/NayanGeneralUtils

import pandas as pd
import bibtexparser as bp
import os
from bradLib.fileHandler import jsonHandler

# Getting the location of this file
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Reading the requirements file and taking fields from it
requirements = jsonHandler.load(os.path.join(__location__, "requirements.json"))
fields = list(requirements["article"].keys())

# Generate an empty bib csv
def generateCSV(fields = fields):
    """Generate an empty csv with a given list of headers"""
    
    df = pd.DataFrame(columns=fields, index=None)
    df.to_csv("emptyBibliography.csv", index=False)
    
    
def cleanEmptyDict(dict):
    """Remove empty entries from a dictionary"""
    
    newDict = {}
    for key in dict.keys():
        if dict[key] != "":
            newDict[key] = str(dict[key])
    return newDict
    
class bibEntry():
    """A class to store single bibtex entries"""
    def __init__(self, dict, ID, type, requirements, fields):
        
        # Initialising an empty dictionaryy
        self.dict = {}
        
        # Storing the ID and type of the entry
        self.ID = ID
        self.type = type
        
        # Storing the fields of the entry in the dictionary
        for field in fields:
            if field in dict.keys():
                self.dict[field] = dict[field]
            else:
                self.dict[field] = ""
        
        # Checking for missing necessary entries
        self.checkMissing(requirements[self.type])
        
        # Storing the ID and type
        self.dict["ID"] = self.ID
        self.dict["ENTRYTYPE"] = self.type
        
        # Cleaning the dictionary
        self.cleanDict = cleanEmptyDict(self.dict)
        
    def checkMissing(self, requirements):
        """Check for missing fields that are required as defined in requirements.json"""
        
        # Checking the required fields
        for field in requirements.keys():
            if (requirements[field] == "required") and (self.dict[field] == ""):
                
                # Printing if a missing field is found and setting its value to "MISSING ENTRY"
                print(f"Entry {self.ID} missing field {field}")
                self.dict[field] = "MISSING ENTRY"
        
class bibliography():
    """A class for storing an entire bibliography"""
    
    # Storing the requirements dict and fields 
    requirements = requirements
    fields = fields
    
    def __init__(self, filePath):
        
        # Storing the file path and extension
        self.filePath = filePath
        self.fileType = filePath[-3:]
        
        # Creating a list to store the entries
        self.entries = []
        
        # Checking the filetype and loading
        if self.fileType == "csv":
            self.fromCSV(filePath)
        elif self.fileType == "bib":
            self.fromBIB(filePath)
        else:
            raise ValueError("File of unknown type entered, accepted types are '.csv' and '.bib' please fix.")
            
    def fromCSV(self, filePath):
        """Load bib data from csv"""
        
        # Reading the csv
        df = pd.read_csv(filePath, na_filter=False)
        
        # Instantiating all entries as bibEntry objects
        for i in range(len(df)):
            entry = df.iloc[i].to_dict()
            
            # Instantiating the entry
            self.instantiateEntry(entry)
    
    def fromBIB(self, filePath):
        """Load bib data from bib"""
        
        # Reading the bib file
        with open(filePath) as file:
            fileStr = file.read()
        
        # Loading as a bibtexparser library
        bib = bp.loads(fileStr)
        
        # Instantiating all entries as bibEntry objects
        for entry in bib.entries:
            
            # Instantiating the entry
            self.instantiateEntry(entry)
            
    def instantiateEntry(self, entry):
        
        # Getting the ID and type
        try: ID = entry.pop("ID")
        except: raise ValueError(f"ID missing, please check input data for entry {entry}")
        try: type = entry.pop("ENTRYTYPE")
        except: raise ValueError(f"Type missing, please check input data for entry {entry}")
        
        # Adding the entry to the list
        self.entries.append(bibEntry(entry, ID, type, self.requirements, self.fields))
        
    def toCSV(self, filePath):
        """Save the bibliography as a csv"""
        
        # The order for the columns
        order = ["ID", "ENTRYTYPE", "title", "author", "year", "journal", "doi", 
                 "url", "number", "pages", "publisher", "address", "annote", 
                 "booktitle", "crossref", "edition", "editor", "howpublished", 
                 "institution", "month", "note", "organization", "school", 
                 "series", "issn", "isbn", "eprint"]
        
        # Instantiating the dataframe and ordering it
        df = pd.DataFrame(columns = self.fields)
        df = df.reindex(order, axis=1)
        
        # Loading the entries into the dataframe
        for entry in self.entries:
            df.loc[len(df)] = entry.cleanDict
        
        # Saving the dataframe as csv
        df.to_csv(filePath, index=False)
    
    def toBIB(self, filePath):
        """Save the bibliography as a bib"""
        
        # Initialising list for storage
        entryList = []
        
        # Storing the entry dictionaries
        for entry in self.entries:
            entryList.append(entry.cleanDict)
        
        # Creating a bibdatabase object to save the bib file
        bib = bp.bibdatabase.BibDatabase()
        bib.entries = entryList
        
        writer = bp.bwriter.BibTexWriter()
        
        with open(filePath, "w") as bibfile:
            bibfile.write(writer.write(bib))
