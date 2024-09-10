import experiments
import rq1_quality

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.patches import PathPatch
import seaborn as sns
import sys
from scipy.stats import mannwhitneyu, rankdata
from math import log10, floor

RENAME = {
    'perfQ': 'performance',
    'reliability': 'reliability',
    '#changes': 'cost',
    'pas': 'pas',
}


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

    ax.set_xlabel(r'{}:     MWU p: {}   -   $\hat{{A}}_{{12}}$: {} ({})'.format(
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
                        s=100, alpha=.9, zorder=2-i, legend=False)
    set_ticker(scat_ax, .05, .1)
    scat_ax.set_xlim([-.1, .45])
    scat_ax.set_ylim([0, 1])
    scat_ax.set_ylabel('reliability', fontsize=20)
    scat_ax.set_xlabel('performance', fontsize=20)

    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=16)

    fig_save(fig, 'rq1-qi-2dscatter-{}'.format(title))

def prepare_obj_values(experiments, pareto=False):
    if pareto:
        obj_values = [e.get_pareto()[RENAME.keys()] for e in experiments]
    else:
        obj_values = [e.get_objectives_df()[RENAME.keys()] for e in experiments]
    obj_values = [df.assign(experiment=exp.short_name)
                  for df, exp in zip(obj_values, experiments)]
    obj_values = pd.concat(obj_values)
    obj_values = obj_values.rename(columns=RENAME)
    obj_values['reliability'] = -obj_values['reliability']
    obj_values['performance'] = -obj_values['performance']
    return obj_values

def pairplot(experiments, title):
    obj_values = prepare_obj_values(experiments, pareto=True)
    labels = [e.short_name for e in experiments]
    colors = ['dimgrey', 'lightgrey']

    fig = sns.pairplot(obj_values, vars=RENAME.values(),
                       hue='experiment', palette=colors, diag_kind='hist',
                       plot_kws={'s': 100, 'edgecolor': 'k', 'alpha': .9},
                       diag_kws={'bins': 24, 'edgecolor': 'k', 'multiple': 'dodge'},
                       height=4, aspect=1)

    # Increase font size
    for ax in fig.axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)

    # Legend above the top axis
    fig._legend.remove()
    fig.fig.legend(fig._legend.legendHandles, labels, loc='upper center',
                   bbox_to_anchor=(.5, 1.03), ncol=2, fontsize=20)

    fig.savefig('plots/rq1-obj-pairplot-{}.pdf'.format(title))

def boxplots(experiments, title):
    obj_values = prepare_obj_values(experiments)
    labels = [e.short_name for e in experiments]
    colors = ['dimgrey', 'lightgrey']

    fig, axs = plt.subplots(1, len(RENAME), figsize=(4*len(RENAME), 5))

    for i, obj in enumerate(RENAME.values()):
        g = sns.boxplot(obj_values, hue='experiment', y=obj, ax=axs[i],
                    palette=colors, linewidth=2, width=.6)
        axs[i].set_xlabel(obj, fontsize=20)
        axs[i].set_ylabel('')
        axs[i].tick_params(axis='both', which='major', labelsize=16)
        axs[i].get_legend().remove()

    # Create legend handles using patches
    handles = [PathPatch([], facecolor=colors[i], edgecolor='k') for i in range(2)]
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(.5, 1.03), ncol=2, fontsize=20)
    fig.subplots_adjust(top=.9)

    plt.savefig('plots/rq1-obj-boxplot-{}.pdf'.format(title))
    fig.clear()
    plt.close(fig)

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
        int_name = interaction.short_name.replace(' ', '_')
        boxplots(
            [interaction, baseline],
            '{}-{}'.format(casestudy, int_name))
        pairplot(
            [interaction, baseline],
            '{}-{}'.format(casestudy, int_name))
        plot_2d_scatter(
            [interaction, baseline],
            ['perfQ', 'reliability'], qi_df, ['HV', 'IGDPlus', 'Eps'],
            '{}-{}'.format(casestudy, int_name))
