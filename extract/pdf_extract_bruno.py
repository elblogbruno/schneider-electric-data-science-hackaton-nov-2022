import numpy as np
import textract
import pandas as pd
from tika import parser
import os

def get_pdf_train_dataset():

    columNames = [
        "facilityName",
        "FacilityInspireID",
        "CountryName",
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
        "DAYS WITH FOG",
        "REPORTER NAME",
        "CITY ID"
    ]

    data_files = []

    files = os.listdir('train/train6')
    for file in files:
        raw = parser.from_file('train/train6/'+file)
        
        to_process = raw['content'].split('\n')
        
        lines = []
        
        for line in to_process:
            lines.append(line)

        data_cleaned = [x for x in lines if x != '']

        # remove all blank spaces 
        data_splited = [x.split(':') for x in data_cleaned]

        print(data_splited)

        for element in  data_splited:
            if len(element) == 2:
                print(element)
                for column in columNames:
                    if column in element:
                        data = element.split(column)
                        data_splited.remove(element)
                        data_splited.append(data)
                        break

if __name__ == '__main__':
    get_pdf_train_dataset()