import pandas as pd
from tika import parser
import os
from math import log10, floor

def find_exp(number) -> int:
    base10 = log10(abs(number))
    return abs(floor(base10))

def get_pdf_train_dataset():

    columNames = [
    "EPRTRSectorCode",
    "eprtrSectorName",
    "FacilityInspireID",
    "CITY",
    "CITY_ID",
    "targetRealase",
    "pollutant",
    "DAY",
    "MONTH",
    "YEAR",
    "COUNTRY",
    "CONTINENT",
    "max_wind_speed",
    "avg_wind_speed",
    "min_wind_speed",
    "max_temp",
    "avg_temp",
    "min_temp",
    "DAYS FOG",
    "FACILITY NAME",
    "REPORTER NAME"]

    columNamesv2 = [ "facilityName",
        "FacilityInspireID",
        "countryName",
        "CONTINENT",
        "City",
        "EPRTRSectorCode",
        "eprtrSectorName",
        "targetRelease",
        "pollutant",
        "DAY",
        "MONTH",
        "reportingYear",
        "max_wind_speed",
        "min_wind_speed",
        "avg_wind_speed",
        "max_temp",
        "min_temp",
        "avg_temp",
        "DAY WITH FOGS",
        "REPORTER NAME",
        "CITY ID"]

    data_files = []

    files = os.listdir('train/train6')
    for file in files:
        raw = parser.from_file('train/train6/'+file)
        to_process = raw['content'].split('\n')
        lines = []
        for line in to_process:
            lines.append(line)
        data_cleaned = [x for x in lines if x != '']
        data_splited = [x.split(':') for x in data_cleaned]

        # BORRAR ESPACIOS INICIALES
        for ind, lista in enumerate(data_splited):
            for ind2, elem in enumerate(lista):
                data_splited[ind][ind2] = elem.strip()

        # ELIMINAR CORCHETES
        new_data = []
        for ind, lista in enumerate(data_splited):
            for ind2, elem in enumerate(lista):
                new_data.append(elem)

        #ARREGLOS ESPECIFICOS
        cont = "CONTINENT"
        new_data[8] = new_data[8].replace(cont, "")
        new_data[8] = new_data[8].strip()
        new_data.insert(9, cont)

        epr = "eprtrSectorName"
        new_data[14] = new_data[14].replace(epr, "")
        new_data[14] = new_data[14].strip()
        new_data.insert(15, epr)

        splt = new_data[25].split()
        new_data[25] = splt[0]
        new_data.insert(26, splt[1])

        splt = new_data[27].split()
        new_data[27] = splt[0]
        new_data.insert(28, splt[1])

        splt = new_data[32].split()
        new_data[32] = splt[0]
        new_data.insert(33, splt[1])

        splt = new_data[34].split()
        new_data[34] = splt[0]
        new_data.insert(35, splt[1])

        splt = new_data[-7].split()
        new_data[-7] = splt[0]
        new_data.insert(-6, splt[1])

        splt = new_data[-1].split()
        new_data[-1] = splt[0]
        new_data.append(splt[1])

        splt = new_data[20].split()
        new_data[20] = splt[0]
        new_data.insert(21, splt[1])

        splt = new_data[-10].split()
        new_data[-10] = splt[0]
        new_data.insert(-9, splt[1])



        data_np = []
        for ind, elem in enumerate(new_data):
            for name in columNames:
                if name == elem:
                    # print("yes")
                    data_np.append(new_data[ind + 1])

        data_files.append(data_np)
        dataF = pd.DataFrame(data_files)
        dataF.columns = columNamesv2

        columns_to_fix  = ['max_temp', 'min_temp', 'avg_temp', 'max_wind_speed', 'min_wind_speed', 'avg_wind_speed']

        for col in columns_to_fix:
            val = dataF[col].to_string(index=False)

            # get val type 
            # print(type(val))    

            if ',' in val:
                val = val.replace(',', '.')

            if '\n' in val:
                val = val.split('\n')[0]
            
            val = float(val)
            # print(val)

            exp = find_exp(val)

            if exp > 0:
                val = val / 10 ** exp
                # print(val)
                dataF[col] = val

    return dataF