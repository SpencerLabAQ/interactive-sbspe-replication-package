import experiments
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

import pprint
import pandas as pd
import pfevaluator

from pymoo.config import Config
Config.warnings['not_compiled'] = False

def print_metrics(metrics):
    print("Pareto front metrics:")
    pp = pprint.PrettyPrinter()
    pp.pprint(metrics)

def normalize_minmax(pf, min_range, max_range):
    pf_ = pf.copy()
    for c in pf.columns:
        pf_[c] = (pf[c] - min_range[c]) / (max_range[c] - min_range[c])
    return pf_

def normalize_fronts(pf, ref_pf, objectives):
    # Min and max ranges of each objective in the reference front
    min_ref = {c:ref_pf[c].min() for c in ref_pf.columns}
    max_ref = {c:ref_pf[c].max() for c in ref_pf.columns}

    # Normalize the reference front between 0 and 1
    ref_pf_norm = normalize_minmax(ref_pf[objectives], min_ref, max_ref)

    # Normalize the pareto front between min and max ranges
    # of each objective in the reference front
    pf_norm = normalize_minmax(pf[objectives], min_ref, max_ref)

    return pf_norm, ref_pf_norm

def compute_hv(pf_df, objectives=None, normalize=True):
    ind = HV([1] * len(objectives))
    return ind(pf_df.values)

def compute_igdplus(pf_df, reference_front, objectives=None, normalize=True):
    ind = IGDPlus(reference_front[objectives].values)
    return ind(pf_df.values)

def compute_epsilon(my_front, reference_front, objectives=None):
    return max(
      [min(
        [max([s2[k] - s1[k] for k in range(len(s2))]) for s2 in my_front.values]
       )
       for s1 in reference_front[objectives].values]
    )

def compute_nps(objectives=None, my_front=None):
    nps = len(set([tuple(x) for x in my_front.values]))
    return nps

def compute_all_metrics(reference_front=None, objectives=None, normalize=True,
                        my_front=None):
    dict_metrics = dict()
    dict_metrics['HV'] = compute_hv(my_front, objectives, normalize)
    dict_metrics['IGDPlus'] = compute_igdplus(my_front, reference_front,
                                              objectives, normalize)
    dict_metrics['Eps'] = compute_epsilon(my_front, reference_front, objectives)
    dict_metrics['NPS'] = compute_nps(objectives, my_front)
    return dict_metrics

def compute_metrics(pf, ref_pf, objectives, experiment):
    # Normalize both the reference and this front
    pf_norm, ref_pf_norm = normalize_fronts(pf, ref_pf, objectives)

    metrics = compute_all_metrics(
        objectives=objectives,
        reference_front=ref_pf_norm[objectives],
        normalize=False,
        my_front=pf_norm[objectives])
    metrics['experiment'] = experiment
    return metrics

def read_and_compute_metrics(exp, ref_pf=None):
    pf = exp.get_pareto()

    # Against itself if no reference is provided
    if ref_pf is None:
        print('No reference front provided. Using the front of the experiment')
        ref_pf = pf.copy()

    # Metrics for the super pareto front
    metrics = compute_metrics(pf, ref_pf, exp.objectives, exp.short_name)

    # Metrics for the pareto front of each run
    by_run = runs_metrics(exp, ref_pf, exp.short_name)

    return metrics, by_run

def runs_metrics(exp, ref_pf, point):
    metrics = []
    for source in exp.objectives_df['source'].unique():
        df = exp.objectives_df
        df = df[df['source'] == source]
        run_metrics = compute_metrics(df, ref_pf, exp.objectives, point)
        metrics.append(run_metrics)
    return pd.DataFrame(metrics)

def compute_ref_pf(casestudy):
    pfs = [exp.get_pareto() for exp in getattr(experiments, casestudy)]
    pfs = pd.concat(pfs)

    # Compute the reference pareto front
    ref_pf = non_dominated(pfs)
    print('Number of solutions in the reference front:', len(ref_pf))
    return ref_pf

def non_dominated(pf):
    # Remove duplicates
    pf = pf.drop_duplicates()

    # Remove dominated solutions
    non_dom = []
    for i, row in pf.iterrows():
        dominated = False
        for j, row2 in pf.iterrows():
            if i == j:
                continue
            if all(row <= row2):
                dominated = True
                break
        if not dominated:
            non_dom.append(row)
    return pd.DataFrame(non_dom)

def compare_metrics(casestudy):
    cmp_metrics = []
    cmp_metrics_byrun = []

    # ref_pf = None
    ref_pf = compute_ref_pf(casestudy)
    for exp in getattr(experiments, casestudy):
        metrics, by_run = read_and_compute_metrics(exp, ref_pf=ref_pf)
        #if exp.short_name == 'reference 1000':
        #    ref_pf = exp.get_pareto()
        cmp_metrics.append(metrics)
        cmp_metrics_byrun.append(by_run)

    # Comparison of metrics
    print('\n{} metrics'.format(casestudy))
    df = pd.DataFrame(cmp_metrics)
    print(df)

    # Standard deviation (rounded to 3 decimals) of the metrics by run
    print('\n{} standard deviation'.format(casestudy))
    df = pd.concat(cmp_metrics_byrun)
    print(df.groupby('experiment').std().round(3))

    # Save metrics by run
    if isinstance(cmp_metrics_byrun[0], pd.DataFrame):
        pd.concat(cmp_metrics_byrun)\
                .to_csv('{}_metrics_by_run.csv'.format(casestudy))


if __name__ == "__main__":
    compare_metrics('ttbs')
    compare_metrics('ccm')
