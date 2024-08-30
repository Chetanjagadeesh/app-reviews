import pandas as pd
import numpy as np
import re


def clean_dataframe(data):

    data.columns = [col.lower() for col in data.columns]

    data['at'] = pd.to_datetime(data['at']).apply(lambda x: x.isoformat() if pd.notnull(x) else None)
    data['repliedat'] = pd.to_datetime(data['repliedat']).apply(lambda x: x.isoformat() if pd.notnull(x) else None)


    data_types = {
    'reviewid': 'string',
    'username': 'string',
    'userimage': 'string',
    'content': 'string',
    'score': 'int',
    'thumbsupcount': 'int',
    'reviewcreatedversion': 'string',
    'at': 'datetime64[ns]',
    'replycontent': 'string',
    'repliedat': 'datetime64[ns]',
    'appversion': 'string'}

    for column, dtype in data_types.items():
        if dtype.startswith('datetime'):
            data[column] = pd.to_datetime(data[column], errors='coerce')
        elif dtype == 'int':
            data[column] = pd.to_numeric(data[column], errors='coerce').fillna(0).astype(int)
        else:
            data[column] = data[column].astype(dtype)
    
    return data 