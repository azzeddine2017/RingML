# File: examples/chess_utils.ring
# Description: Helper functions for Chess Data processing

load "stdlib.ring"
load "csvlib.ring"

# Map string labels to Class Indices (0-17)
aClassMap = [
    "draw", "zero", "one", "two", "three", "four", "five", "six", "seven", 
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", 
    "fifteen", "sixteen"
]

func getFileIndex cChar
    # Convert 'a'->1, 'b'->2 ... 'h'->8
    cChar = lower(cChar)
    nAscii = ascii(cChar)
    # ascii('a') is 97
    return nAscii - 96

func normalizeBoard aInputs
    # Inputs are in range 1-8. We normalize to 0-1 for the Neural Net
    # aInputs is a list of 6 numbers
    aNorm = list(6)
    for i=1 to 6
        aNorm[i] = aInputs[i] / 8.0 
    next
    return aNorm

func getLabelIndex cString
    nIdx = find(aClassMap, cString)
    if nIdx = 0 
        raise("Unknown label: " + cString) 
    ok
    return nIdx # Returns 1-based index (Ring), we will adjust to 1-based for OneHot

func getLabelName nIndex
    if nIndex < 1 or nIndex > len(aClassMap) return "Unknown" ok
    return aClassMap[nIndex]

func readCSV cFile
    if ! fexists(cFile)
        raise("File " + cFile + " doesn't exist!")
    ok
    return CSV2List( read(cFile) )
