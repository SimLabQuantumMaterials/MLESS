import numpy as np
import csv

elementAtomicNumbers = {
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71
}

mixtureRatios = [0.75, 0.625, 0.5, 0.375, 0.25]


def readData(filepath):
    res = []
    with open(filepath) as file:
        # skip first row
        file.__next__()

        for row in file:
            rowSplit = row.split()

            element1, element2 = elementAtomicNumbers[rowSplit[0]], elementAtomicNumbers[rowSplit[1]]

            for i in range(len(mixtureRatios)):
                x = mixtureRatios[i]
                HE = float(rowSplit[2 + i])

                res.append([element1, element2, x, HE])

    return np.array(res)


def readCSVData(filepath, separator=";", material="monazite", bulkshear=False, Volume=False):
    res = []
    with open(filepath) as file:
        # skip first row
        file.__next__()
        csv_reader = csv.reader(file, delimiter=separator)

        for row in csv_reader:
            # element = elementAtomicNumbers[row[0]]
            Z = int(row[1])
            atomicMass = float(row[2])
            if material == "xenotime":
                #print("read data for xenozite")
                R = float(row[3])
            else:
                R = float(row[4])
            IP2 = float(row[5])
            IP3 = float(row[6])
            electronegativity = float(row[7])
            E = float(row[8])
            nuclearCharge = float(row[9])
            density = float(row[10])
            bulkmodulus = float(row[11])
            shearmodulus = float(row[12])

            Vol = float(row[13])
                
            if bulkshear:
                if Volume:
                    res.append([Z, atomicMass, IP2,  E, nuclearCharge, bulkmodulus, shearmodulus, electronegativity, IP3, Vol, R])
                else:
                    res.append([Z, atomicMass, IP2, E, nuclearCharge, bulkmodulus, shearmodulus, electronegativity, IP3, R])                    
            else:
                if Volume:
                    res.append([Z, atomicMass, IP2,  E, nuclearCharge, electronegativity, IP3, Vol, R])
                else:
                    res.append([Z, atomicMass, IP2,  E, nuclearCharge, electronegativity, IP3, R])
                
                
    return np.array(res)


def readCSVData_simplified(filepath, separator=";",  material="monazite", R=False):
    res = []
    with open(filepath) as file:
        # skip first row
        file.__next__()
        csv_reader = csv.reader(file, delimiter=separator)

        for row in csv_reader:
            if R:
                if material == "xenotime":
                    R = float(row[3])
                else:
                    R = float(row[4])
                E = float(row[8])
                res.append([E, R])
            else:
                E = float(row[8])
                Vol = float(row[13])
                res.append([E, Vol])
                
                
    return np.array(res)

