import experiments

def dominates_or_equal(sol_a, sol_b):
    """ Check if solution sol_a dominates or is equal to solution sol_b """
    return all(x <= y for x, y in zip(sol_a, sol_b))

def coverage_metric(df_a, df_b):
    """ Calculate the coverage metric C(a, b) """
    set_a = [tuple(x) for x in df_a.to_numpy()]
    set_b = [tuple(x) for x in df_b.to_numpy()]
    count = 0
    for sol_b in set_b:
        if any(dominates_or_equal(sol_a, sol_b) for sol_a in set_a):
            count += 1
    return count / len(set_b)

def load_experiments_and_compute_coverage(casestudy):
    """ Load the experiments and compute the coverage metric """
    baseline = [e for e in getattr(experiments, casestudy)
                if e.short_name == 'baseline 100'][0]
    exps = [e for e in getattr(experiments, casestudy)
            if e.short_name.startswith('2nd step')]
    for exp in exps:
        print(f'{casestudy} - {exp.short_name} - C(exp, base): ' + \
              f'{coverage_metric(exp.get_pareto(), baseline.get_pareto())}')
        print(f'{casestudy} - {exp.short_name} - C(base, exp): ' + \
              f'{coverage_metric(baseline.get_pareto(), exp.get_pareto())}')

if __name__ == "__main__":
    load_experiments_and_compute_coverage('ttbs')
    load_experiments_and_compute_coverage('ccm')
