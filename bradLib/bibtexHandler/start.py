
import bibtexHandler

"""Start the input-output for CLI use"""

# Taking the input and output file paths
IN = str(input("Input file name: "))
OUT = str(input("Output file name: "))

# Instantiating a bibliography object
bib = bibtexHandler.bibliography(IN)

# Saving the file as a csv or bib depending on the output filename
if OUT[-3:] == "csv":
    bib.toCSV(OUT)
elif OUT[-3:] == "bib":
    bib.toBIB(OUT)
else:
    raise ValueError("Something went wrong, please check the input filenames")