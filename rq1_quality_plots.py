import experiments
import rq1_quality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import sys
from scipy.stats import mannwhitneyu, rankdata
from math import log10, floor


### Stats

def a12(df, qi, experiments):
    if len(experiments) > 2:
        print('Max 2 experiments can be compared')
        exit(1)
    sample1 = df[df['experiment'] == experiments[0]][qi]
    sample2 = df[df['experiment'] == experiments[1]][qi]
    m = len(sample1)
    n = len(sample2)
    r1 = sum(rankdata(np.concatenate([sample1,sample2]))[:m])
    return (2 * r1 - m * (m + 1)) / (2 * n * m)

def interpret_a12(a12):

    # M. Hess and J. Kromrey, 2004
    # "Robust confidence intervals for effect sizes"
    levels = [0.147, 0.33, 0.474]

    magnitude = ["N", "S", "M", "L"]
    scaled_a12 = (a12 - 0.5) * 2
    return magnitude[np.searchsorted(levels, abs(scaled_a12))]

def mwu(df, qi, experiments):
    if len(experiments) > 2:
        print('Max 2 experiments can be compared')
        exit(1)
    _, p = mannwhitneyu(
            df[df['experiment'] == experiments[0]][qi],
            df[df['experiment'] == experiments[1]][qi])
    return p
###

def check_args():
    if len(sys.argv) < 3:
        print('Usage: python {} <CSV with quality indicators> <case study name>'\
                .format(sys.argv[0]))
        exit(0)

def fig_save(fig, name):
    plt.tight_layout()
    plt.savefig('plots/{}.pdf'.format(name))
    fig.clear()
    plt.close(fig)

def set_ticker(ax, xstep, ystep):
    xloc = plticker.MultipleLocator(base=xstep)
    yloc = plticker.MultipleLocator(base=ystep)
    ax.xaxis.set_major_locator(xloc)
    ax.yaxis.set_major_locator(yloc)

def pretty_sci(n):
    if int(n) == n:
        return int(n)
    if n < .001:
        exp = floor(log10(n))
        round_n = round(n / 10**exp, 3)
        return r'${} \times 10^{{{}}}$'.format(round_n, exp)
    return round(n, 3)

def superscript(n):
    inp = str(n)
    out = ''
    if str(n)[0] == '-':
        out = '⁻'
        inp = inp[1:]
    return out + "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c)-ord('0')] for c in inp])

def subplot_dist(ax, qi_df, qi, labels, colors):
    # Dist
    rename = {'HV': 'Hypervolume', 'IGDPlus': 'IGD+', 'Eps': 'Epsilon'}
    sns.histplot(data=qi_df, x=qi, hue='experiment', hue_order=labels, ax=ax,
                 bins=24, multiple='dodge', palette=colors, edgecolor='k')
    ax.set_ylabel('count')


    # MWU
    mwu_p = mwu(qi_df, qi, labels)

    # A12
    a12_ = a12(qi_df, qi, labels)
    a12_m = interpret_a12(a12_)

    ax.set_xlabel('{}:     MWU p: {}   -   A12: {} ({})'.format(
        rename[qi], pretty_sci(mwu_p), pretty_sci(a12_), a12_m),
                  fontsize=20)
    ax.set_ylabel('count', fontsize=20)

def plot_2d_scatter(experiments, objectives, qi_df, qis, title):

    fig, axs = plt.subplots(4, 1,
            gridspec_kw={'height_ratios': [1, 1, 1, 4]}, figsize=(8, 10))

    colors = ['dimgrey', 'lightgrey']
    paretos = [e.get_pareto(invert=True)[objectives] for e in experiments]
    labels = [e.short_name for e in experiments]

    # Distributions subplots
    for i, qi in enumerate(qis):
        df = qi_df[qi_df['experiment'].isin(labels)]
        subplot_dist(axs[i], df, qi, labels, colors)

    # Legend above the top axis
    axs[0].legend(axs[0].get_legend().legend_handles, labels, loc='upper center',
                 bbox_to_anchor=(.5, 1.8), ncol=2, fontsize=20)
    # Remove all the others
    for ax in axs[1:3]:
        ax.legend([], [])

    # Scatter
    scat_ax = axs[3]
    for i, pf in enumerate(paretos):
        sns.scatterplot(data=pf, x=objectives[0], y=objectives[1], ax=scat_ax,
                        label=labels[i], color=colors[i], edgecolor='k',
                        s=75, alpha=.9, zorder=2-i, legend=False)
    set_ticker(scat_ax, .05, .1)
    scat_ax.set_xlim([-.1, .45])
    scat_ax.set_ylim([0, 1])
    scat_ax.set_ylabel('reliability', fontsize=20)
    scat_ax.set_xlabel('performance', fontsize=20)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=16)

    fig_save(fig, 'rq1-qi-2dscatter-{}'.format(title))

def get_exps_by_name(name, exps):
    return [exp for exp in exps if exp.short_name.startswith(name)]

if __name__ == "__main__":
    check_args()
    qi_csv = sys.argv[1]
    casestudy = sys.argv[2]
    qi_df = pd.read_csv(qi_csv)

    baseline =  get_exps_by_name('baseline 100', getattr(experiments, casestudy))[0]
    interactions = get_exps_by_name('2nd step 50 ', getattr(experiments, casestudy))

    for interaction in interactions:
        plot_2d_scatter(
            [interaction, baseline],
            ['perfQ', 'reliability'], qi_df, ['HV', 'IGDPlus', 'Eps'],
            '{}-{}'.format(casestudy, interaction.short_name.replace(' ', '_')))
