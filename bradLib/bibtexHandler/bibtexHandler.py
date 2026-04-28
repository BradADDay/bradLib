
import pandas as pd
import bibtexparser as bp
import os
from bradLib.fileHandler import jsonHandler

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

requirements = jsonHandler.load(os.path.join(__location__, "requirements.json"))
keys = list(requirements["article"].keys())
    
def cleanEmptyDict(dict):
    
    newDict = {}
    
    for key in dict.keys():
        
        if dict[key] != "":
            newDict[key] = str(dict[key])
    
    return newDict
    
class bibEntry():
    
    def __init__(self, dict, ID, type, requirements, keys):
        
        self.dict = {}
        self.ID = ID
        self.type = type
        
        for key in keys:
            try:
                self.dict[key] = dict[key]
            except:
                self.dict[key] = ""
        
        self.checkMissing(requirements[self.type])
        
        self.dict["ID"] = self.ID
        self.dict["ENTRYTYPE"] = self.type
        
        self.cleanDict = cleanEmptyDict(self.dict)
        
    def checkMissing(self, requirements):
        
        for field in requirements.keys():
            
            if (requirements[field] == "required") and (self.dict[field] == ""):
                
                print(f"Entry {self.ID} missing field {field}")
                self.dict[field] = "MISSING ENTRY"
        
class bibliography():
    
    requirements = requirements
    keys = keys
    
    def __init__(self, filePath):
        
        self.filePath = filePath
        self.fileType = filePath[-3:]
        self.entries = []
        
        if self.fileType == "csv":
            self.fromCSV(filePath)
        elif self.fileType == "bib":
            self.fromBIB(filePath)
        else:
            raise ValueError("Unkown file type entered, please fix.")
            
    def fromCSV(self, filePath):
        
        df = pd.read_csv(filePath, na_filter=False)
        
        for i in range(len(df)):
            entry = df.iloc[i].to_dict()
            key = entry.pop("key")
            type = entry.pop("type")
            self.entries.append(bibEntry(entry, key, type, self.requirements, self.keys))
    
    def fromBIB(self, filePath):
        
        with open(filePath) as file:
            fileStr = file.read()
        
        bib = bp.loads(fileStr)
        
        for entry in bib.entries:
            key = entry.pop("ID")
            type = entry.pop("ENTRYTYPE")
            self.entries.append(bibEntry(entry, key, type, self.requirements, self.keys))
            
    def toCSV(self, filePath):
        
        order = ["ID", "ENTRYTYPE", "title", "author", "year", "journal", "doi", 
                 "url", "number", "pages", "publisher", "address", "annote", 
                 "booktitle", "crossref", "edition", "editor", "howpublished", 
                 "institution", "month", "note", "organization", "school", 
                 "series", "issn", "isbn", "eprint"]
        
        df = pd.DataFrame(columns = self.keys)
        df = df.reindex(order, axis=1)
        
        for entry in self.entries:
            
            df.loc[len(df)] = entry.cleanDict
        
        df.to_csv(filePath, index=False)
    
    def toBIB(self, filePath):
        
        entryList = []
        
        for entry in self.entries:
            entryList.append(entry.cleanDict)
        
        bib = bp.bibdatabase.BibDatabase()
        bib.entries = entryList
        
        writer = bp.bwriter.BibTexWriter()
        
        with open(filePath, "w") as bibfile:
            bibfile.write(writer.write(bib))

def generateCSV(keys = keys):
    df = pd.DataFrame(columns=keys, index=None)
    df.to_csv("emptyBibliography.csv", index=False)
    
def startIO():
    IN = str(input("Input file name: "))
    choice = input("(0) To csv\n(1) To bib\nEnter Selection: ")
    OUT = str(input("Output file name: "))
    
    bib = bibliography(IN)
    
    if int(choice) == 0:
        bib.toCSV(OUT)
    elif int(choice) == 1:
        bib.toBIB(OUT)
    else:
        print("Something went wrong")