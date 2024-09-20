import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

QI = {
    'HV': r'$HV$ ($\uparrow$)',
    'IGDPlus': r'$IGD\text{+}$ ($\downarrow$)',
    'Eps': r'$EP$ ($\downarrow$)',
}

experiments = {
    'reference 1000': '$Reference_{1000}$',
    'baseline 100': '$Baseline_{100}$',
    '2nd step 50 c258': '$Interactive^{2nd (c258)}_{50}$',
    '2nd step 50 c223': '$Interactive^{2nd (c223)}_{50}$',
    '2nd step 50 c317': '$Interactive^{2nd (c317)}_{50}$',
    '2nd step 50 c358': '$Interactive^{2nd (c358)}_{50}$',
}

def plot_boxplot(df, qi, app):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 1.6))
    sns.boxplot(x=qi, y="experiment", data=df, width=0.8, linewidth=1.5,
                fliersize=2, color='lightgray', ax=ax)
    ax.set_xlabel(QI[qi], fontsize=14)
    filename = 'plots/rq1_qi_boxplots_{}_{}.pdf'.format(app, qi)
    plt.savefig(filename, bbox_inches='tight')
    print("Saved to", filename)

def check_args():
    if len(sys.argv) != 2:
        print("Usage: python {} <path_to_csv>".format(sys.argv[0]))
        sys.exit(1)

if __name__ == "__main__":
    check_args()
    path = sys.argv[1]
    app = path.split('_')[0]
    df = pd.read_csv(path)
    df['experiment'] = df['experiment'].apply(lambda x: experiments[x])

    for qi in QI:
        plot_boxplot(df, qi, app)
