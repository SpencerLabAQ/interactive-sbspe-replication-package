import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from glob import glob

DATA = 'zips'
DATA_TMP = 'datasets'
EXPERIMENTS = [
    'nsgaii-train-ticket-1000-eval',
    'nsgaii-train-ticket-sbspe-100-eval-it-0',
    'nsgaii-train-ticket-sbspe-50-eval-it-1-l-2-centroid-258',
    'nsgaii-train-ticket-sbspe-50-eval-it-1-l-2-centroid-223',
    'nsgaii-cocome-1000-eval',
    'nsgaii-cocome-sbspe-100-eval-it-0',
    'nsgaii-cocome-sbspe-50-eval-it-1-l-2-centroid-317',
    'nsgaii-cocome-sbspe-50-eval-it-1-l-2-centroid-358',
]
OBJECTIVES = [
    'perfQ',
    '#changes',
    'pas',
    'reliability',
]

def prepare_data():
    zips = Path(DATA)
    data_dir = Path(DATA_TMP)

    if not data_dir.exists() and \
       not data_dir.is_dir() and \
       zips.exists() and \
       zips.is_dir():
         data_dir.mkdir()
         archives = ['{}/{}.zip'.format(DATA, exp) for exp in EXPERIMENTS]
         for arc in archives:
             with ZipFile(arc, 'r') as zObject:
                 zObject.extractall(path=DATA_TMP)

def read_data():
    data = []
    for exp in EXPERIMENTS:
        for f in glob('{}/{}/objectives/*.csv'.format(DATA_TMP, exp)):
            df = pd.read_csv(f, usecols=OBJECTIVES)
            df['usecase'] = exp.split('-')[1]
            data.append(df)
    df = pd.concat(data)
    df['perfQ'] = -df['perfQ']
    df['reliability'] = -df['reliability']

    return df

def correlation_matrix(df):
    return df.corr().round(4)

if __name__ == "__main__":
    prepare_data()
    df = read_data()

    print('##### Overall correlation matrix')
    print(correlation_matrix(df[OBJECTIVES]).to_csv())

    for usecase in ['train', 'cocome']:
        print('##### Correlation matrix for usecase {}'.format(usecase))
        print(correlation_matrix(df[df['usecase'] == usecase][OBJECTIVES]).to_csv())
