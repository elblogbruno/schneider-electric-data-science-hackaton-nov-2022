import pandas as pd
import requests

API_ENDPOINT = "http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/"
TRAIN_DATASET =  ['first', 'second', 'third']


def _extract_json(dataset):
    """
    Extracts the json from the API endpoint
    """
    response = requests.get(API_ENDPOINT + dataset)
    json_data = response.json()
    return json_data

def _process_json(json_data):
    """
    Processes the json data
    """
    # parse JSON as dataframe with headers as column names
    df = pd.DataFrame.from_dict(json_data)
    return df

"""
Loads the JSON dataset from the API endpoint and returns a pandas dataframe
    Returns:  A pandas dataframe
"""
def get_json_train_dataset(dataset_name):
    """
    Gets the json dataset from the API endpoint
    """

    json_data = _extract_json(dataset_name)
    json_data = _process_json(json_data)

    
    return json_data