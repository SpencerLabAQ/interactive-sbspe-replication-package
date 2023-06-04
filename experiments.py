import pandas as pd
import pfevaluator
from pathlib import Path
from glob import glob
from zipfile import ZipFile
from architecturespaceanalyzer import ArchitectureSpaceAnalyzer

DATA = 'zips'
DATA_TMP = 'datasets'

class Experiment:
    def __init__(self, name, short_name, length, num_objs):
        self.name = name
        self.short_name = short_name
        self.length = length
        self.num_objs = num_objs
        self.objectives = self.get_objectives()
        self.objectives_df = self.get_objectives_df()

    def get_objectives(self):
        objectives = ['perfQ', 'reliability', '#changes']
        if self.num_objs == 4:
            objectives.append('pas')
        return objectives

    def get_objectives_df(self):
        space = ArchitectureSpaceAnalyzer()
        space.initialize_dataset(self.name)
        space.read_file_batch(1, 31, length=self.length, arguments=2,
                              option='all', add_source=True)
        df = space.objectives_df
        df['perfQ'] = -df['perfQ']
        df['reliability'] = -df['reliability']
        return df

    def get_pareto(self, invert=False):
        pf_values = pfevaluator.find_reference_front(
                self.objectives_df[self.objectives].values)
        pf = pd.DataFrame(pf_values, columns=self.objectives)
        if invert:
            pf['perfQ'] = -pf['perfQ']
            pf['reliability'] = -pf['reliability']
        return pf


#####
# TTBS
#####
ttbs = [
    Experiment('nsgaii-train-ticket-1000-eval',
               'reference 1000', 4, 4),
    Experiment('nsgaii-train-ticket-sbspe-100-eval-it-0',
               'baseline 100', 4, 4),
    Experiment('nsgaii-train-ticket-sbspe-50-eval-it-1-l-2-centroid-258',
               '2nd step 50 c258', 2, 4),
    Experiment('nsgaii-train-ticket-sbspe-50-eval-it-1-l-2-centroid-223',
               '2nd step 50 c223', 2, 4),
]

#####
# CoCOME
#####
ccm = [
    Experiment('nsgaii-cocome-1000-eval',
               'reference 1000', 4, 4),
    Experiment('nsgaii-cocome-sbspe-100-eval-it-0',
               'baseline 100', 4, 4),
    Experiment('nsgaii-cocome-sbspe-50-eval-it-1-l-2-centroid-317',
               '2nd step 50 c317', 2, 4),
    Experiment('nsgaii-cocome-sbspe-50-eval-it-1-l-2-centroid-358',
               '2nd step 50 c358', 2, 4),
]

# Prepare data
zips = Path(DATA)
data_dir = Path(DATA_TMP)

# Extract the archives if necessary
if not data_dir.exists() and \
   not data_dir.is_dir() and \
   zips.exists() and \
   zips.is_dir():
    archives = ['{}/{}.zip'.format(DATA, exp.name) for exp in ttbs + ccm]
    data_dir.mkdir()
    for arc in archives:
        with ZipFile(arc, 'r') as zObject:
            zObject.extractall(path=DATA_TMP)
