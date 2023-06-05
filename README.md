#### Replication package for the paper:

*How do human interactions impact multi-objective optimization of software architecture refactoring?*

---

#### How to generate the tables and figures in the paper
Initialize the python execution environment:
```shell
git clone https://github.com/SpencerLabAQ/replication-package__interactive-search-based-software-performance.git
cd replication-package__interactive-search-based-software-performance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

##### RQ1: quality of solutions
Compute the quality indicators and generate Table 4:
```shell
python rq1_quality.py
```

The previous command will also generate two CSV files (`ttbs_metrics_by_run.csv` and `ccm_metrics_by_run.csv`) which contain a run by run computation of the quality indicators, and are used to generate the plots of RQ1.

Generate the plots in Figures 4 and 5:
```shell
# Train Ticket case study
python rq1_quality_plots.py ttbs_metrics_by_run.csv ttbs

# CoCOME case study
python rq1_quality_plots.py ccm_metrics_by_run.csv ccm
```
The resulting plots will be saved in the [plots](plots) folder.

##### RQ2: architectural differences
Execute notebooks `rq2-cocome.ipynb` or `rq2-trainticket.ipynb` to generate the charts for Figures 7 and 8. 

Among other computations, the notebooks above will generate several CSV and DOT files under the [datasets](datasets) folder.

Execute notebook `tree-scripts.ipynb` to generate the charts for Figure 6.

The datasets needed by the notebooks are automatically loaded from a default folder.

##### RQ3: covered design and solution spaces
Execute notebooks `rq3-cocome.ipynb` or `rq3-trainticket.ipynb` to generate the charts for Figures 9 and 10. 

Among other computations, the notebooks above will generate several CSV and DOT files under the [datasets](datasets) folder.

Execute notebook `tree-scripts.ipynb` to generate the charts for Figures 11 and 12.

The datasets needed by the notebooks are automatically loaded from a default folder.

#### Experiments

The experiments in the paper were performed using [EASIER](http://sealabtools.di.univaq.it/EASIER/), which is available in a different repository:

[https://github.com/SEALABQualityGroup/EASIER](https://github.com/SEALABQualityGroup/EASIER)

All the data gathered during such experiments is provided here, in the [zips](zips) folder.
