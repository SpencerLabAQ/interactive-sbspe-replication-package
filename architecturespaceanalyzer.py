import re

from glob import glob
from os.path import basename

import pandas as pd
import requests

from bs4 import BeautifulSoup

#import gdown
import os
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from natsort import natsorted, ns
import math
from scipy.spatial.distance import pdist, squareform
from distinctipy import distinctipy
#from fastDamerauLevenshtein import damerauLevenshtein
from collections import Counter, OrderedDict

import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder

import string

from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.features import ParallelCoordinates
from yellowbrick.features import RadViz

import plotly.graph_objects as go
import plotly.offline as pyo

import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
from scipy.stats import entropy
from scipy import spatial
from functools import reduce

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import matplotlib

from pandas.plotting import parallel_coordinates

from collections import defaultdict

from pca import pca

import altair as alt

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from difflib import SequenceMatcher

import zss

from pymoo.visualization.petal import Petal
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights

from pymoo.indicators.gd import GD
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

#from sklearn.manifold import TSNE
from openTSNE import TSNE

#import hdbscan


###########################################

class ArchitectureSpaceAnalyzer:

  def __init__(self, project=None, objectives=None):
    # All global (class-level) variables for a given project
    self.ALL_OBJECTIVES = ['perfQ', 'reliability', '#changes', 'pas']
    self.ALL_ARGUMENTS = ['operation', 'target', 'to', 'where']
    self.INITIAL_SOLUTION = None #{'solID': 0, 'perfQ': 0.0, 'reliability': 0.7630625563279512, '#changes': 0, 'pas': 12}
    self.ALL_REFACTORINGS = []
    self.OBJECTIVES_PATH = ''
    self.OBJECTIVES_FILES = ''
    self.REFACTIONS_PATH = ''
    self.REFACTIONS_FILES = ''
    self.PROJECT_NAME = ''
    self.FILE_INDEX = -1
    self.FILE_DESCRIPTION = ''

    # These are the labels to discretize each of the features in a 5-point scale
    self.PERFORMANCE_LABELS = ['very-slow','slow', 'average', 'fast', 'very-fast'] # Is this absolute 0..1?
    self.RELIABILITY_LABELS = ['unreliable','minimally-reliable', 'average', 'reliable','very-reliable'] # This is absolute 0..1
    self.PAS_LABELS = ['very-few','few', 'average', 'some','many']
    self.CHANGES_LABELS = ['very-few','few', 'average', 'some','many']
    self.PERFORMANCE_LIMITS = (0,1.0)
    self.RELIABILITY_LIMITS = (0,1.0)
    self.CHANGES_LIMITS = (None,None)
    self.PAS_LIMITS = (None,None)

    self.CLUSTERS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf','#d3d3d3']
    self.CLUSTERS = None
    self.CLUSTER_LABELS = None
    self.CLUSTERS_PALETTE = None
    self.LABELS_COLORS = None

    # Show interactive 2D charts with the solutions and the clusters
    self.TOOLTIP = ['solID','perfQ', 'reliability', '#changes', 'pas', 'label', 'cluster']

    self.ALPHABET = list(string.ascii_letters) + list(string.digits) 
    self.use_alphabet = True

    self.action_parameters = 0 # These are default values for parsing refactoring actions
    self.sequence_length = 4

    self.objectives_df = None
    self.refactions_df = None
    if objectives is None:
      self.objectives = self.ALL_OBJECTIVES
    else:
      self.objectives = objectives

    self.cluster_labels = None
    self.centroids_df = None
    self.tagged_objectives_df = None
    self.pareto_front_df = None

    self._kmeans_kwargs = {
      "init": "random",
      "n_init": 10,
      "max_iter": 300,
      "random_state": 42
    }

    if project is not None:
      self.initialize_dataset(project)
  

  def set_colors(self, n=None, show_palette=False):
    if n is not None:
    # generate N visually distinct colours
      colors = distinctipy.get_colors(n, rng=3)
      self.CLUSTERS_COLORS = [matplotlib.colors.to_hex(c) for c in colors]
      # display the colours
      if show_palette:
        print("New palette:", n, self.CLUSTERS_COLORS)
        distinctipy.color_swatch(colors)
      return colors
    return None
  

  def set_labels(self, cluster_labels):
    n = len(cluster_labels)
    colors = self.set_colors(n)
    self.LABELS_COLORS = dict()
    for idx, lb in enumerate(cluster_labels):
      self.LABELS_COLORS[lb] = matplotlib.colors.to_hex(colors[idx])

    return self.LABELS_COLORS


  # Load both the objectives and refactoring actions for a given case
  def initialize_dataset(self, project):
    self.PROJECT_NAME = project
    print('project:', self.PROJECT_NAME)

    self.OBJECTIVES_PATH = './datasets/'+self.PROJECT_NAME+'/objectives'
    self.OBJECTIVES_FILES = natsorted([f for f in listdir(self.OBJECTIVES_PATH) if isfile(join(self.OBJECTIVES_PATH, f))], key=lambda y: y.lower())
    print('objective files=', len(self.OBJECTIVES_FILES))

    self.REFACTIONS_PATH = './datasets/'+self.PROJECT_NAME+'/refactoring_actions'
    self.REFACTIONS_FILES = natsorted([f for f in listdir(self.REFACTIONS_PATH) if isfile(join(self.REFACTIONS_PATH, f))], key=lambda y: y.lower())
    print('refactoring actions files=', len(self.REFACTIONS_FILES))

###########################################

  # static and private method
  @staticmethod
  def _flatten_block(df, include_parameters=0):
    row_list =[]  
    solID = None
    for index, row in df.iterrows():
      #print(index,row)
      # Create list for the current row
      if type(include_parameters) == int:
          p = include_parameters
      else: # It should be a dictionary
          p = include_parameters[row.operation]
      
      if p == 1:
          row_list.append(row.operation+'('+row.target+')')
      elif p == 2:
          row_list.append(row.operation+'('+row.target+','+str(row.to)+')')
      elif p == 3:
          row_list.append(row.operation+'('+row.target+','+str(row.to)+','+str(row['where'])+')')
      else:
          row_list.append(row.operation)
      solID = row.solID

    return [solID] + row_list 
    #return flat_df


  # static and private method
  @staticmethod
  def _parse_operations(df, k=4, parameters=0): # Parameters control how much information is used for each operation
    #print(k)
    x = 0 # k is the sequence length
    dfs = []
    for y in range(0,df.shape[0]//k):
      z = k*(y+1)
      #print(x, z)
      flat_df = ArchitectureSpaceAnalyzer._flatten_block(df[x:z], include_parameters=parameters)
      dfs.append(flat_df)
      x = z
      #print(flat_df)
      #print('===')
    
    if k == 4: # TODO: Fix this hack later!
      return pd.DataFrame(dfs, columns=['solID', 'op1', 'op2', 'op3', 'op4'])
    else: # k should be 2
      return pd.DataFrame(dfs, columns=['solID', 'op1', 'op2'])


  # Get only the refactoring actions for a given index
  def read_refactoring_actions(self, idx, length=4, arguments=3, verbose=True):
    self.sequence_length = length
    self.action_parameters = arguments
    if verbose:
      print('loading index:',idx)
    self.FILE_INDEX = idx

    ref_path = self.REFACTIONS_PATH+'/'+self.REFACTIONS_FILES[idx] # These are global variables
    if verbose:
      print(ref_path)

    self.FILE_DESCRIPTION = self.REFACTIONS_FILES[idx][len('VAR'+str(idx+1)+'__'):]

    refacts_df = pd.read_csv(ref_path)
    self.ALL_REFACTORINGS = set(refacts_df['operation'])
    #print(self.ALL_REFACTORINGS)
    refacts_df = ArchitectureSpaceAnalyzer._parse_operations(refacts_df, k=length, parameters=arguments)
    if length == 4: # TODO: Fix this hack later!
      duplicate_refacts = refacts_df[refacts_df.duplicated(['op1', 'op2', 'op3', 'op4'], keep=False)]
    else: # length should be 2
      duplicate_refacts = refacts_df[refacts_df.duplicated(['op1', 'op2'], keep=False)]
    if len(duplicate_refacts.index):
      print("Warning: duplicate rows in refactoring actions!", list(duplicate_refacts['solID']))

    self.refactions_df = refacts_df.copy()
    return refacts_df


  # Get only the objectives for a given index
  def read_objectives(self, idx, invert=True, initial_solution=False, verbose=True):
    if verbose:
      print('loading index:',idx)
    self.FILE_INDEX = idx

    obj_path = self.OBJECTIVES_PATH+'/'+self.OBJECTIVES_FILES[idx] # These are global variables
    if verbose:
      print(obj_path)

    self.FILE_DESCRIPTION = self.OBJECTIVES_FILES[idx][len('FUN'+str(idx+1)+'__'):]

    objs_df = pd.read_csv(obj_path)
    if invert:
      objs_df['perfQ'] = (-1)*objs_df['perfQ']
      objs_df['reliability'] = (-1)*objs_df['reliability']
    if initial_solution:
      objs_df = objs_df.append(self.INITIAL_SOLUTION, ignore_index=True)
      objs_df = objs_df.astype({'solID': int})
    if 'pas' not in objs_df.columns: # TODO: Fix this hack later!
      objs_df['pas'] = 0.0
    objs_df = objs_df[['solID']+self.ALL_OBJECTIVES]
    duplicate_objs = objs_df[objs_df.duplicated(self.ALL_OBJECTIVES, keep=False)]
    if len(duplicate_objs.index):
      print("Warning: duplicate rows in objectives!", list(duplicate_objs['solID']))

    self.objectives_df = objs_df.copy()
    return objs_df


  def read_objectives_refactoring_actions(self, idx, invert=True, length=4, arguments=3, initial_solution=False, verbose=True):

    objs_df = self.read_objectives(idx, invert, initial_solution, verbose)
    refacts_df = self.read_refactoring_actions(idx, arguments=arguments, verbose=verbose, length=length)

    return objs_df, refacts_df


  # Get all the feasible files (objectives, refactoring actions, or both) for a given project name
  # Note: in case of files with formatting problems, they are ignored
  def read_file_batch(self, min, max, invert=True, length=4, arguments=3, initial_solution=False, option='all', add_source=False):
    print("Reading files ...", min, max, option)
    objs_dflist = []
    refactions_dflist = []
    for n in range(min-1,max):
      print('  file',n)
      objs_df = None
      refacts_df = None
      try:
        if option == 'all':
          objs_df, refacts_df = self.read_objectives_refactoring_actions(n, length=length, arguments=arguments, initial_solution=False, verbose=False)
          if add_source:
            objs_df['source'] = str(n+1)+'_'+self.FILE_DESCRIPTION
            refacts_df['source'] = str(n+1)+'_'+self.FILE_DESCRIPTION
        if option == 'objectives':
          objs_df = self.read_objectives(n, initial_solution=False, verbose=False)
          if add_source:
            objs_df['source'] = str(n+1)+'_'+self.FILE_DESCRIPTION
        if option == 'refactions':
          refacts_df = self.read_refactoring_actions(n, length=length, arguments=arguments, verbose=False)
          if add_source:
            refacts_df['source'] = str(n+1)+'_'+self.FILE_DESCRIPTION
      except:
        print("\tProblems loading objectives and/or refactoring actions:",(n+1))
        if option == 'all':
          objs_df = None
          refacts_df = None
        if option == 'objectives':
          objs_df = None
        if option == 'refactions':
          refacts_df = None
      #else:
      #  print("\t"+str(n)+": OK", FILE_DESCRIPTION)
      if objs_df is not None:
        objs_dflist.append(objs_df)
      if refacts_df is not None:
        refactions_dflist.append(refacts_df)
    
    print("done.")

    merged_objectives_df = None
    if len(objs_dflist) > 0:
      merged_objectives_df =  pd.concat(objs_dflist, axis=0)
      merged_objectives_df.reset_index(drop=True, inplace=True)
    merged_refactions_df = None
    if len(refactions_dflist) > 0:
      merged_refactions_df =  pd.concat(refactions_dflist, axis=0)
      merged_refactions_df.reset_index(drop=True, inplace=True)

    if merged_objectives_df is not None:
      self.objectives_df = merged_objectives_df.copy()
    else:
      self.objectives_df = None
    if merged_refactions_df is not None:
      self.refactions_df = merged_refactions_df.copy()
    else:
      self.refactions_df = None
    return merged_objectives_df, merged_refactions_df

###########################################

  # static and private method
  @staticmethod
  def _get_silhouette_scores_for_clusters(df, labels, num_clusters):
    sample_silhouette_values = metrics.silhouette_samples(df, labels)
    means_lst = []
    for lb in range(num_clusters):
      #print(lb)
      c_df = sample_silhouette_values[labels == lb]
      means_lst.append(c_df.mean())
  
    return means_lst


  # Apply K-Means to cluster a dataframe (Pareto front) of numeric values
  def run_kmeans(self, k, kwargs=None, n_pca=None, normalize=True, show_silhouette=False):
    if kwargs is None:
      kwargs = self._kmeans_kwargs
    if normalize:
      sample = StandardScaler().fit_transform(self.objectives_df[self.objectives])
    else:
      sample = self.objectives_df[self.objectives]  

    if n_pca is not None:
      pca = PCA(n_components=n_pca)
      sample_pca = pca.fit_transform(sample)
      #print("Explained PCA variance:", np.sum(pca.explained_variance_ratio_))
      print("PCA components:",len(pca.explained_variance_ratio_), pca.explained_variance_ratio_)
      #plt.scatter(sample_pca[:,0], sample_pca[:,1])
      #plt.show()
      kmeans = KMeans(n_clusters=k, **kwargs).fit(sample_pca)
    else:
      kmeans = KMeans(n_clusters=k, **kwargs).fit(sample)
    labels = kmeans.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Labels for instances:', kmeans.labels_)
    fixed_labels = np.where(kmeans.labels_ < 0, 0, kmeans.labels_)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    classes = set(fixed_labels)
    if len(classes) > 1:
      if n_pca is not None:
        silhouette = metrics.silhouette_score(sample_pca, fixed_labels)
        silhouette_scores = ArchitectureSpaceAnalyzer._get_silhouette_scores_for_clusters(sample_pca, fixed_labels, k)
        print("Individual Silhouette scores:", silhouette_scores) 
      else:
        silhouette = metrics.silhouette_score(sample, fixed_labels)
        silhouette_scores = ArchitectureSpaceAnalyzer._get_silhouette_scores_for_clusters(sample, fixed_labels, k)
        print("Individual Silhouette scores:", silhouette_scores) 
    else:
      silhouette = 0.0
    print("Average Silhouette coefficient: %0.3f" % silhouette)   
    #print("Individual Silhouette scores:", sample_silhouette_values)

    if show_silhouette:
      # Instantiate the visualizer
      visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
      if n_pca is not None:
        visualizer.fit(sample_pca)
      else:    
        visualizer.fit(sample)    
      visualizer.show()       

    return fixed_labels, kmeans, silhouette


  # Apply Aglomerative clustering to a dataframe (Pareto front) of numeric values
  def run_agglomerative(self, k, threshold=200, n_pca=None, normalize=True, show_dendogram=False, archstructure=None):
    if normalize:
      sample = StandardScaler().fit_transform(self.objectives_df[self.objectives])
    else:
      sample = self.objectives_df[self.objectives]  

    if n_pca is not None:
      pca = PCA(n_components=n_pca)
      sample_pca = pca.fit_transform(sample)
      #print("Explained PCA variance:", np.sum(pca.explained_variance_ratio_))
      print("PCA components:",len(pca.explained_variance_ratio_), pca.explained_variance_ratio_)
      #plt.scatter(sample_pca[:,0], sample_pca[:,1])
      #plt.show()
      model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward', connectivity=archstructure, distance_threshold=threshold)
      model.fit(sample_pca)
#      X = sample_pca
    else:
      model = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward', connectivity=archstructure, distance_threshold=threshold)
      model.fit(sample)
#      X = sample
    labels = model.labels_
    print(f"Number of clusters = {1+np.amax(model.labels_)}")

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Labels for instances:', model.labels_)
    fixed_labels = np.where(model.labels_ < 0, 0, model.labels_)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    classes = set(fixed_labels)
    if len(classes) > 1:
      if n_pca is not None:
        silhouette = metrics.silhouette_score(sample_pca, fixed_labels)
        silhouette_scores = ArchitectureSpaceAnalyzer._get_silhouette_scores_for_clusters(sample_pca, fixed_labels, k)
        print("Individual Silhouette scores:", silhouette_scores) 
      else:
        silhouette = metrics.silhouette_score(sample, fixed_labels)
        silhouette_scores = ArchitectureSpaceAnalyzer._get_silhouette_scores_for_clusters(sample, fixed_labels, k)
        print("Individual Silhouette scores:", silhouette_scores) 
    else:
      silhouette = 0.0
    print("Average Silhouette coefficient: %0.3f" % silhouette)   
    #print("Individual Silhouette scores:", sample_silhouette_values) 

    if show_dendogram:
      fig = plt.figure(figsize=(20,10))
      ax = fig.add_subplot(1, 1, 1)
      dendrogram = sch.dendrogram(sch.linkage(sample, method='ward'), ax=ax)   
      plt.show()    

    return fixed_labels, model, silhouette


#  def run_hdbscan(self, epsilon=0.0, min_cluster_size=5, n_pca=None, normalize=True, 
#                  show_scatter=False, show_condensed_tree=False, show_linkage=False):
#    if normalize:
#      sample = StandardScaler().fit_transform(self.objectives_df[self.objectives])
#    else:
#      sample = self.objectives_df[self.objectives]  
#
#    if n_pca is not None:
#      pca = PCA(n_components=n_pca)
#      sample_pca = pca.fit_transform(sample)
#      #print("Explained PCA variance:", np.sum(pca.explained_variance_ratio_))
#      print("PCA components:",len(pca.explained_variance_ratio_), pca.explained_variance_ratio_)
#      #plt.scatter(sample_pca[:,0], sample_pca[:,1])
#      #plt.show()
#      model = hdbscan.HDBSCAN(cluster_selection_epsilon=epsilon, min_cluster_size=min_cluster_size)
#      model.fit(sample_pca)
##      X = sample_pca
#    else:
#      model = hdbscan.HDBSCAN(cluster_selection_epsilon=epsilon, min_cluster_size=min_cluster_size)
#      model.fit(sample)
##      X = sample
#
#    labels = model.labels_
#    print(f"Number of clusters = {1+np.amax(model.labels_)}")
#
#    # Number of clusters in labels, ignoring noise if present.
#    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#    n_noise_ = list(labels).count(-1)
#
#    print('Labels for instances:', model.labels_)
#    fixed_labels = np.where(model.labels_ < 0, 0, model.labels_)
#
#    print("Estimated number of clusters: %d" % n_clusters_)
#    print("Estimated number of noise points: %d" % n_noise_)
#    classes = set(fixed_labels)
#    k = model.labels_.max()
#    if len(classes) > 1:
#      if n_pca is not None:
#        silhouette = metrics.silhouette_score(sample_pca, fixed_labels)
#        silhouette_scores = ArchitectureSpaceAnalyzer._get_silhouette_scores_for_clusters(sample_pca, fixed_labels, k)
#        print("Individual Silhouette scores:", silhouette_scores) 
#      else:
#        silhouette = metrics.silhouette_score(sample, fixed_labels)
#        silhouette_scores = ArchitectureSpaceAnalyzer._get_silhouette_scores_for_clusters(sample, fixed_labels, k)
#        print("Individual Silhouette scores:", silhouette_scores) 
#    else:
#      silhouette = 0.0
#    print("Average Silhouette coefficient: %0.3f" % silhouette)   
#    #print("Individual Silhouette scores:", sample_silhouette_values)   
#
#    if show_scatter:
#      fig = plt.figure(figsize=(10,10))
#      color_palette = sns.color_palette('deep', k+1)
#      cluster_colors = [color_palette[x] if x >= 0
#                  else (0.5, 0.5, 0.5)
#                  for x in model.labels_]
#      cluster_member_colors = [sns.desaturate(x, p) for x, p in
#                         zip(cluster_colors, model.probabilities_)]
#      pca2 = PCA(n_components=2)
#      sample_pca2 = pca2.fit_transform(sample)
#      plt.scatter(sample_pca2[:, 0], sample_pca2[:, 1], s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
#      plt.xlabel('PCA1 '+str(round(100*pca2.explained_variance_ratio_[0],2))+"%")
#      plt.ylabel('PCA2 '+str(round(100*pca2.explained_variance_ratio_[1],2))+"%")
#      plt.show()
#    
#    if show_condensed_tree:
#      color_palette = sns.color_palette('deep', k+1)
#      model.condensed_tree_.plot(select_clusters=True, selection_palette=color_palette)
#
#    if show_linkage:
#      model.single_linkage_tree_.plot()
#
#    return fixed_labels, model, silhouette


###########################################

  def show_parallel_plot(self, cluster_labels=None, title=None, size=(600,400), normalize=False):
    sample = self.objectives_df[self.objectives] 
    if title is None:
      title = self.PROJECT_NAME

    features = sample.columns 
    if cluster_labels is None:
      classes = None
      palette_clusters = sns.color_palette(None, 1) 
    else:
      classes = set(cluster_labels)
      palette_clusters = sns.color_palette("husl",len(classes))

    # Trying to visualize the different clusters 
    n = None
    if normalize:
      n = 'standard'
      visualizer = ParallelCoordinates(classes=classes, features=features, fast=False, 
                                       title=title, size=size, colors=palette_clusters, normalize=n)
    if cluster_labels is None:
      cluster_labels = [-1]*(sample.shape[0])
      #print(cluster_labels)
    visualizer.fit_transform(sample, cluster_labels)
    visualizer.show(legend=None)

    #visualizer.ax.legend().set_visible(False)
    plt.show()

    return palette_clusters


  # Apply PCA on the dataframe
  def _apply_pca(self, df, n_pca=2, normalize=True):

    if normalize:
      sample = StandardScaler().fit_transform(df)
    else:
      sample = df  
    pca_ = PCA(n_components=n_pca)
    sample_pca = pca_.fit_transform(sample)
    print("PCA components:",len(pca_.explained_variance_ratio_), pca_.explained_variance_ratio_)

    return sample_pca, pca_


  # Apply t-SNE on the dataframe
  def _apply_tsne(self, df, n_tsne=2, normalize=True):
    if normalize:
      sample = StandardScaler().fit_transform(df)
    else:
      sample = df  
    tsne_embedding = TSNE(n_components=n_tsne, random_state=1).fit(sample)
    #sample_tsne = tsne_.fit_transform(sample)

    return tsne_embedding


  # Show the clusters after a 2D t-SNE reduction 
  def show_clusters_tsne(self, cluster_labels=None, palette=None, title=None, size=(600,400), tsne=None):
    df = self.objectives_df[self.ALL_OBJECTIVES]
    tsne_ = tsne
    if tsne_ is None:
      tsne_sample = self._apply_tsne(df)
      tsne_ = tsne_sample
    else:
      print("Reusing t-SNE")
      tsne_sample = tsne_.transform(StandardScaler().fit_transform(df.values))
    print(len(df.columns), "features", list(df.columns))
    if title is None:
      title = space.PROJECT_NAME

    # Show the 2D representation
    plt.figure(figsize=(size[0]/60,size[1]/60))
    if cluster_labels is not None:
      print(len(set(cluster_labels)), "clusters")
    else:
      print("no clusters")
    if palette is None:
      plt.scatter(tsne_sample[:, 0], tsne_sample[:, 1], c='blue', s=50, alpha=0.7, cmap='viridis')
    else:
      cluster_colors = [palette[c] for c in cluster_labels]
      plt.scatter(tsne_sample[:, 0], tsne_sample[:, 1], c=cluster_colors, s=50, alpha=0.7, cmap='viridis')
 
    # title and labels
    plt.title(title, fontsize=16)

    plt.show()
    #return palette
    return tsne_


  # Show the clusters after a 2D PCA reduction 
  def show_clusters_pca_2d(self, cluster_labels=None, palette=None, title=None, size=(600,400), pca=None):
    df = self.objectives_df[self.ALL_OBJECTIVES]
    pca_ = pca
    if pca_ is None:
        sample_pca, pca_ = self._apply_pca(df)
    else:
        print("Reusing PCA")
        sample_pca = pca_.transform(StandardScaler().fit_transform(df.values))
    print(len(df.columns), "features", list(df.columns))
    if title is None:
      title = self.PROJECT_NAME

    # Show the 2D representation
    plt.figure(figsize=(size[0]/60,size[1]/60))
    if cluster_labels is not None:
      print(len(set(cluster_labels)), "clusters")
    else:
      print("no clusters")
    if palette is None:
      plt.scatter(sample_pca[:, 0], sample_pca[:, 1], c='blue', s=50, alpha=0.7, cmap='viridis')
    else:
      cluster_colors = [palette[c] for c in cluster_labels]
      plt.scatter(sample_pca[:, 0], sample_pca[:, 1], c=cluster_colors, s=50, alpha=0.7, cmap='viridis')
 
    # title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('PCA1 '+str(round(100*pca_.explained_variance_ratio_[0],2))+"%")
    plt.ylabel('PCA2 '+str(round(100*pca_.explained_variance_ratio_[1],2))+"%")

    plt.show()
    #return palette
    return pca_


  # Alternative PCA implementation (supporting both 2D and 3D projections)
  def show_clusters_pca(self, cluster_labels=None, n_components=3, kind='2D', size=(18,10), title=None):
    if title is None:
      title = self.PROJECT_NAME
    X = self.objectives_df[self.ALL_OBJECTIVES] 
    if cluster_labels is not None:
      y = cluster_labels 
    else:
      y = np.array([-1]*X.shape[0])

    #model = pca(n_components=0.95)
    model = pca(n_components=n_components)

    results = model.fit_transform(X)
    print("Top features:", results['topfeat'])
    # Cumulative explained variance
    print("Cumulative explained variance:", model.results['explained_var'])
    # Explained variance per PC
    print("Explained variance:", model.results['variance_ratio']) 

    # Make 3D Plot
    if kind == '3D':
      fig, ax = model.biplot3d(n_feat=6, PC=[0,1,2], y=y, label=False, figsize=size, title=title)
    else:
      # Make 2D plot
      fig, ax = model.biplot(n_feat=6, PC=[0,1], y=y, label=False, figsize=size, title=title)
    ax.title.set_fontsize(15)
    ax.title.set_fontweight('bold')
    fig.show()

    #plt.show() 


  def show_radviz(self, cluster_labels, palette=None, title=None, size=(800, 600)):
    if title is None:
      title = self.PROJECT_NAME
    X = self.objectives_df[self.objectives] 
    if cluster_labels is not None:
      y = cluster_labels 
    else:
      y = np.array([-1]*X.shape[0])
    
    clusters = set(y)
    # Instantiate the visualizer
    visualizer = RadViz(classes=clusters, colors=palette, title=title, size=size)
    visualizer.fit_transform(X, y)  # Fit the data to the visualizer
    #visualizer.transform(X)        # Transform the data
    visualizer.show() 
    plt.show()


  def show_density_plot(self, objectives=None, pca=False, kind='kde', bins=5, 
                        normalize=False, rug=True, levels=10,
                        size=(12,10), vminmax=(None,None), title=None, xlim=None, ylim=None):
    if title is None:
      title = self.PROJECT_NAME
    
    pca_ = None
    df = self.objectives_df[self.ALL_OBJECTIVES].copy()
    if (pca is True) and (objectives is None):
      objectives = ['pca1', 'pca2']
      pca_data, pca_ = self._apply_pca(df, normalize=True)
      df = pd.DataFrame(pca_data, columns=objectives)
    elif (pca is not None) and (objectives is None):
      objectives = ['pca1', 'pca2']
      print("Reusing PCA transformation ...")
      print("PCA components:",len(pca.explained_variance_ratio_), pca.explained_variance_ratio_)
      pca_data = pca.transform(StandardScaler().fit_transform(df))
      df = pd.DataFrame(pca_data, columns=objectives)
    elif objectives is None: # PCA is false
      objectives = self.objectives #self.ALL_OBJECTIVES[0:2]

    de = None
    if not pca:
      de = round(self.compute_density_entropy(objectives=objectives, bins=bins),2)
      title = title + ' [de=' + str(de) +']'#str(objectives)

    if normalize or (kind == 'bins'):
      if (xlim is not None) and (ylim is not None):
        print("Scaling with:", xlim, ylim)
        df[objectives[0]] = (df[objectives[0]] - xlim[0]) / (xlim[1] - xlim[0])
        df[objectives[1]] = (df[objectives[1]] - ylim[0]) / (xlim[1] - ylim[0])
        data = df
      else:
        data = MinMaxScaler().fit_transform(df[objectives])
        data = pd.DataFrame(data, columns=objectives)
    else:
      data = df

    if kind == 'kde':
#      kde_plot = sns.kdeplot(data=data, x=objectives[0], y=objectives[1], 
#                         cmap=sns.color_palette("rocket_r", as_cmap=True),
#                         vmin=vminmax[0], vmax=vminmax[1], cbar=True,
#                         levels=levels, shade_lowest=False, shade=True)
      kde_plot = sns.displot(data=data, x=objectives[0], y=objectives[1], 
                         cmap=sns.color_palette("rocket_r", as_cmap=True),
                         vmin=vminmax[0], vmax=vminmax[1], cbar=True,
                         rug=rug, fill=True, kind="kde",height=6, aspect=11.7/8.27)
    
    if kind == 'bins':
      fig,ax = plt.subplots(figsize=size)
      bins_plot = sns.histplot(data=data, x=objectives[0], y=objectives[1], # TODO: This needs to be replaces by  a heatmap
                             cmap=sns.color_palette("rocket_r", as_cmap=True),
                             vmin=vminmax[0], vmax=vminmax[1], cbar=True,
                             stat='probability', bins=bins, binrange=(xlim,ylim))
      #ax.xaxis.set_major_locator(ticker.MultipleLocator(1/5))
      #ax.yaxis.set_major_locator(ticker.MultipleLocator(1/5))
  
    if normalize or (kind == 'bins'):
      plt.xlim([0,1])
      plt.ylim([0,1])
    elif (not normalize) and (xlim is not None) and (ylim is not None):
      plt.xlim(xlim)
      plt.ylim(ylim)

    plt.title(title, fontsize=16)
    plt.show()
    
    return pca_

  def _get_minmax(self, obj):
    if obj == '#changes':
      return self.CHANGES_LIMITS
    if obj == 'pas':
      return self.PAS_LIMITS
    if obj == 'perfQ':
      return self.PERFORMANCE_LIMITS
    if obj == 'reliability':
      return self.RELIABILITY_LIMITS
    return None, None


  def _normalize_objectives(self, objectives=None):
    if objectives is None:
      objectives = self.objectives

    df = self.objectives_df[objectives].copy()
    for obj in objectives:
      minobj, maxobj = self._get_minmax(obj)
      if (minobj is None) and (maxobj is None):
        minobj = df[obj].min()
        maxobj = df[obj].max()
      df[obj] = (df[obj] - minobj) / (maxobj - minobj)

    return df

  # This is a measure of the homogeneity of space, in terms of the density of solutions. 
  # If closer to 1, the space is "bumpy" (valleys and peaks). If closer to 0, the space is flat.
  def compute_density_entropy(self, objectives=None, normalize=True, bins=10):
    if objectives is None:
      objectives = self.objectives
    if normalize:
      data = self._normalize_objectives(objectives).values #MinMaxScaler().fit_transform(self.objectives_df[objectives])
      spacing = np.linspace(0,1,bins+1)
      bin_edges = [tuple(x) for x in [spacing]*len(objectives)]
      #print(n_tuples)
    else:
      data = self.objectives_df[objectives].values
      bin_edges = bins

    histogram_nd, _ = np.histogramdd(data, bins=bin_edges, density=False)
    array_1d = histogram_nd.ravel() 
    #print("Total:", np.sum(array_1d))
    #print("Array size:", array_1d.size)
    array_1d = array_1d / np.sum(array_1d)
    #print(np.sum(array_1d))

    return entropy(array_1d, base=array_1d.size)


###########################################

  @staticmethod 
  def _get_near_centroid(df, cluster): # Select the point with the smallest distance to a given theoretical centroid (point)
    objectives = list(df.columns)
    objectives.remove('cluster')
    if 'label' in objectives:
      objectives.remove('label')
    if 'prototype' in objectives:
      objectives.remove('prototype')
    #print(objectives)
    points_of_cluster = df[df['cluster']==cluster]
    centroid_of_cluster = np.mean(points_of_cluster[objectives], axis=0) 
    #print("centroid=", centroid_of_cluster)

    #print([centroid_of_cluster])
    #print(points_of_cluster[objectives])
    pos = np.argmax(-distance.cdist([centroid_of_cluster], points_of_cluster[objectives], metric='euclidean'))
    idx = points_of_cluster.index[pos]
    #print(pos,idx)
    #print("Returning a real centroid ...")
    return idx, points_of_cluster.loc[idx]


  @staticmethod 
  def _get_centroids(df, cluster_choice=[]): # Get the real centroids for each cluster
    centroids = []
    #clusters = set(df['cluster'])
    if len(cluster_choice) == 0:
      clusters = set(df['cluster'])
    else:
      clusters = cluster_choice
    for c in clusters:
      pos, _ = ArchitectureSpaceAnalyzer._get_near_centroid(df, c) # Get the real centroids for each cluster
      centroids.append(pos)
  
    return centroids
  

  # Generate the (real) cluster centroids as prototypes for each cluster
  @staticmethod 
  def _add_cluster_prototypes(df, protos=None, cluster_choice=[]):
    df['prototype'] = False
    if protos is None:
      protos = ArchitectureSpaceAnalyzer._get_centroids(df, cluster_choice)
      print("Configuring (real) cluster centroids as prototypes ...", protos)
    else:
      print("Using alternatives as prototypes ...", protos)

    for p in protos:
      df.at[p, 'prototype'] = True

    return df


  @staticmethod
  def _get_dist_bins(col, n): # Split the values of the column into n buckets, according to the distribution of values
    unique_values = sorted((set(col.values)))
    #print(unique_values)
    #sorted(set(pd.qcut(unique_values, 5)))
    return sorted(set(pd.qcut(unique_values, n).map(lambda x: x.left))) + [max(unique_values)]


  def _get_bins(col, n, min_max=(None,None)):
    unique_values = sorted((set(col.values)))
    if min_max == (None,None):
      min_x = np.min(unique_values)
      max_x = np.max(unique_values)
    else:
      print("Using predefined limits",min_max)
      min_x = min_max[0]
      max_x = min_max[1]
    min_x = min_x - 0.1
    max_x = max_x + 0.1
    delta = (max_x - min_x)/n
    #print(min_x, max_x, delta)
    return ([min_x+i*delta for i in range(0,n)] + [max_x])


  @staticmethod
  def _get_fixed_bins(col, n): # Split the values of the column into n fixed buckets (i.e., same size)
    min = col.min()
    #if min < 0:
    #  min = 0
    max = col.max()
    #if max > 1:
    #  max = 1
    _, fixed_bins = pd.cut([min,max], bins=5, retbins=True)
    print(min,max)
    return fixed_bins


  @staticmethod
  def _extract_cluster_labels(df, clusters):
    dict_cluster_label = {}
    for c in clusters:
      lb = df[df['cluster'] == c]['label'].values[0]
      dict_cluster_label[c] = lb
    
    n_labels = len(set(dict_cluster_label.values()))
    if n_labels < len(clusters):
      print("Warning: some clusters have the same labels, so they might not be distinguishable!")
    
    return dict_cluster_label

  def _update_cluster_information(self):
    print("Updating clustering information ...")
    self.CLUSTERS = set(self.centroids_df['cluster'])
    #print(self.CLUSTERS)
    self.CLUSTER_LABELS = ArchitectureSpaceAnalyzer._extract_cluster_labels(self.centroids_df, self.CLUSTERS)
    #print(self.CLUSTER_LABELS)
    if self.LABELS_COLORS is None:
      self.CLUSTERS_PALETTE = sns.color_palette(self.CLUSTERS_COLORS[0:len(self.CLUSTERS)])
    else:
      color_list = [self.LABELS_COLORS[lb] for lb in self.CLUSTER_LABELS.values()]
      self.CLUSTERS_PALETTE = sns.color_palette(color_list)
    # TODO: Try to get always the same palette, based on the labels assigned to each cluster

    return self.CLUSTERS_PALETTE


  def assign_cluster_labels(self, labels):
    self.cluster_labels, n, self.centroids_df, self.tagged_objectives_df = self.get_cluster_labels(labels)
    # Warning: Labels are relative to the range of values of each objective
    # print(n, cluster_labels)
    self.tagged_objectives_df['solID'] = self.objectives_df['solID']

    self.centroids_df['cluster'] = self.cluster_labels.keys()
    self.centroids_df['label'] = self.cluster_labels.values()
    self._update_cluster_information()

    return self.centroids_df


  # Discretize the points and centroids of the clusters with a 5-point scale
  # TODO: Review so that the labels are only assigned depending on the selected objectives
  def get_cluster_labels(self, labels):
    classes = set(labels) #set(df['cluster'].values)

    print("Discretization of features:")
    print("===")
    sample_df = self.objectives_df[self.ALL_OBJECTIVES].copy()
    sample_df['cluster'] = labels

    # Performance
    if 'perfQ' in self.objectives:
      c = sample_df['perfQ']
      bins = ArchitectureSpaceAnalyzer._get_bins(c,5,self.PERFORMANCE_LIMITS)
      print("perfQ -->", bins)
      #perf_lbs = pd.cut(c, bins=get_bins(c,5), labels=performance_labels) #performance_labels[::-1]) # Reversing the list of labels?
      perf_lbs = pd.cut(c, bins=bins, labels=self.PERFORMANCE_LABELS) 
    else:
      perf_lbs = pd.Series(['-']*sample_df.shape[0]) # This is a void column/series 
    print(self.PERFORMANCE_LABELS)
    print(set(perf_lbs))
    print("===")

    # Number of changes
    if '#changes' in self.objectives:
      c = sample_df['#changes']
      bins = ArchitectureSpaceAnalyzer._get_bins(c,5,self.CHANGES_LIMITS)
      print("#changes -->", bins)
      #changes_lbs = pd.cut(c, bins=get_bins(c,5), labels=changes_labels) #changes_labels[::-1]) # Reversing the list of labels?
      changes_lbs = pd.cut(c, bins=bins, labels=self.CHANGES_LABELS) #changes_labels[::-1]) # Reversing the list of labels?
    else:
      changes_lbs = pd.Series(['-']*sample_df.shape[0]) # This is a void column/series 
    print(self.CHANGES_LABELS)
    print(set(changes_lbs))
    print("===")

    # Performance antipatterns
    if 'pas' in self.objectives:
      c = sample_df['pas']
      bins = ArchitectureSpaceAnalyzer._get_bins(c,5,self.PAS_LIMITS)
      print("pas -->", bins)
      #pas_lbs = pd.cut(c, bins=get_bins(c,5), labels=pas_labels) #pas_labels[::-1]) # Reversing the list of labels?
      pas_lbs = pd.cut(c, bins=bins, labels=self.PAS_LABELS) #pas_labels[::-1]) # Reversing the list of labels?
    else:
      pas_lbs = pd.Series(['-']*sample_df.shape[0]) # This is a void column/series 
    print(self.PAS_LABELS)
    print(set(pas_lbs))
    print("===")

    # Reliability
    if 'reliability' in self.objectives:
      c = sample_df['reliability']
      bins = ArchitectureSpaceAnalyzer._get_bins(c,5,self.RELIABILITY_LIMITS)
      print("reliability -->",bins)
      #rel_lbs = pd.cut(c, bins=get_bins(c,5), labels=reliability_labels) # Reverse not needed here
      rel_lbs = pd.cut(c, bins=bins, labels=self.RELIABILITY_LABELS) # Reverse not needed here
    else:
      rel_lbs = pd.Series(['-']*sample_df.shape[0]) # This is a void column/series 
    print(self.RELIABILITY_LABELS)
    print(set(rel_lbs))
    print("===")

    centroids_ids = ArchitectureSpaceAnalyzer._get_centroids(sample_df) # These are the "real" centroids (not the theoretical ones)
    centroids = sample_df.loc[centroids_ids]
    #print(centroids)

    combined_labels = []
    for x, y, z, w in zip(perf_lbs,rel_lbs, changes_lbs,pas_lbs):
      merged_label = x + " / " + y + " / " + z + " / " + w
      #print(merged_label)
      combined_labels.append(merged_label)
  
    #print(len(combined_labels))
    sample_df['label'] = combined_labels #pd.Series(combined_labels)
    if sample_df['label'].isna().any():
      print("Series has NaN!")
    #print(sample_df.shape, "all labels:", sample_df)

    labels = dict()
    for c, ck in zip(centroids_ids, classes):
      labels[ck] = sample_df.loc[c]['label']
  
    n_labels = len(set(labels.values()))
    if n_labels < len(classes):
      print("Warning: some clusters have the same labels, so they might not be distinguishable!")
  
    # Computing the centroids as "reference points" of each cluster
    centroids_df = pd.DataFrame(centroids, columns=self.ALL_OBJECTIVES)

    return labels, n_labels, centroids_df, sample_df


  def describe_cluster_labels(self, objectives=[]):
    sample_df = self.objectives_df[self.ALL_OBJECTIVES]
    n = 5
    # Performance
    c = sample_df['perfQ']
    perf_vals = ArchitectureSpaceAnalyzer._get_bins(c,n,self.PERFORMANCE_LIMITS)
    # Number of changes
    c = sample_df['#changes']
    change_vals = ArchitectureSpaceAnalyzer._get_bins(c,n,self.CHANGES_LIMITS)
    # Performance antipatterns
    c = sample_df['pas']
    pas_vals = ArchitectureSpaceAnalyzer._get_bins(c,n,self.PAS_LIMITS)
    # Reliablility
    c = sample_df['reliability']
    rel_vals = ArchitectureSpaceAnalyzer._get_bins(c,n,self.RELIABILITY_LIMITS)

    list_items = []
    # Performance 
    for i in range(0,n):
        dict_i = dict()
        dict_i['objective'] = 'perfQ'
        dict_i['min'] = perf_vals[i]
        dict_i['max'] = perf_vals[i+1]
        dict_i['unit'] = 'unit'
        dict_i['label'] = self.PERFORMANCE_LABELS[i]
        list_items.append(dict_i)
  
    # Cost of changes
    for i in range(0,n):
        dict_i = dict()
        dict_i['objective'] = '#changes'
        dict_i['min'] = change_vals[i]
        dict_i['max'] = change_vals[i+1]
        dict_i['unit'] = 'unit'
        dict_i['label'] = self.CHANGES_LABELS[i]
        list_items.append(dict_i)

    # Performance antipatterns
    for i in range(0,n):
        dict_i = dict()
        dict_i['objective'] = 'pas'
        dict_i['min'] = pas_vals[i]
        dict_i['max'] = pas_vals[i+1]
        dict_i['unit'] = 'unit'
        dict_i['label'] = self.PAS_LABELS[i]
        list_items.append(dict_i)
  
    # Reliability
    for i in range(0,n):
        dict_i = dict()
        dict_i['objective'] = 'reliability'
        dict_i['min'] = rel_vals[i]
        dict_i['max'] = rel_vals[i+1]
        dict_i['unit'] = 'unit'
        dict_i['label'] = self.RELIABILITY_LABELS[i]
        list_items.append(dict_i)

    all_labels =  pd.DataFrame(list_items)
    if len(objectives) > 0:
      return all_labels[all_labels['objective'].isin(objectives)]
    else:
      return all_labels


  def show_cluster_labels_distribution(self, size=(10,10)):
    dict_labels = self.tagged_objectives_df['label'].value_counts(normalize=True).to_dict()
    if self.LABELS_COLORS is not None:
      for k in self.LABELS_COLORS.keys():
        if k not in dict_labels.keys():
          dict_labels[k] = 0.0

    sorted_dict = {k:v for k,v in sorted(dict_labels.items())}

    fig = plt.figure(figsize=size)
    plt.barh(list(sorted_dict.keys()), list(sorted_dict.values()), color='orange')
    #plt.xticks(rotation=60, ha='right')
    plt.title('Frequency of cluster labels ('+self.PROJECT_NAME+')')
    plt.xlim([0,1.0])
    plt.show()

    return sorted_dict
    
    
  # It requires the CLUSTERS and CLUSTERS_COLORS variables to be set!
  def show_objective_space(self, source=None, prototypes=None, title=None, cluster_choice=[]):

    tooltip = self.TOOLTIP
    if title is None:
      title = self.PROJECT_NAME

    if source is None:
      source = self.tagged_objectives_df
    if len(cluster_choice) > 0:
      source_reduced = source[source['cluster'].isin(cluster_choice)] 
      source = source_reduced.copy()
    source = ArchitectureSpaceAnalyzer._add_cluster_prototypes(source, prototypes, cluster_choice)

    click = alt.selection_point()
    selection_clusters = alt.selection_point(fields=['cluster'])
    #selection_prototypes = alt.selection_multi(fields=['prototype'])
    #input_dropdown = alt.binding_select(options=['True','False'], name='Reference points (only)')
    #selection_prototypes = alt.selection_single(fields=['prototype'], bind=input_dropdown)

    clabels = [self.CLUSTER_LABELS[idx] for idx in self.CLUSTERS]
    hexa_colors = self.CLUSTERS_COLORS
    if self.LABELS_COLORS is not None:
      hexa_colors = [self.LABELS_COLORS[lb] for lb in clabels]
    color_clusters = alt.condition(selection_clusters|click,
                      #alt.Color('cluster:N', legend=None, scale=alt.Scale(scheme='category10')),
                      alt.Color('cluster:O', legend=None, scale=alt.Scale(domain=list(self.CLUSTERS), range=hexa_colors)),
                      alt.value('transparent') #https://vega.github.io/vega/docs/schemes/#reference
                      )

    legend_clusters = alt.Chart(source).mark_point(size=150).encode(
        y=alt.Y('cluster:N', axis=alt.Axis(orient='right')),
        #y=alt.Y('cluster_label', axis=alt.Axis(orient='right')),
        color=color_clusters
    ).add_params(selection_clusters)

    #color_prototypes = alt.condition(selection_prototypes,
    #                    alt.value('black'),
    #                    alt.value('transparent') #https://vega.github.io/vega/docs/schemes/#reference
    #                    )
  
    #legend_prototypes = alt.Chart(source).mark_point(size=150).encode(
    #      y=alt.Y('prototype:O', axis=alt.Axis(orient='right', title='reference points')), #color=alt.value('black')
    #      color=color_prototypes
    #).add_selection(selection_prototypes)

    extra = 0.2
    perf_min = source['perfQ'].min()-extra
    perf_max = source['perfQ'].max()+extra
    changes_min = source['#changes'].min()-extra
    changes_max = source['#changes'].max()+extra
    rel_min = source['reliability'].min()-extra
    rel_max = source['reliability'].max()+extra
    pas_min = source['pas'].min()-extra
    pas_max = source['pas'].max()+extra

    perf_threshold_x = alt.Chart(pd.DataFrame({'x': [perf_max-extra]})).mark_rule(color='gray').encode(x='x')
    pas_threshold_x = alt.Chart(pd.DataFrame({'x': [pas_min+extra]})).mark_rule(color='gray').encode(x='x')

    changes_threshold_y = alt.Chart(pd.DataFrame({'y': [changes_min+extra]})).mark_rule(color='gray').encode(y='y')
    rel_threshold_y = alt.Chart(pd.DataFrame({'y': [rel_max-extra]})).mark_rule(color='gray').encode(y='y')
    perf_threshold_y = alt.Chart(pd.DataFrame({'y': [perf_max-extra]})).mark_rule(color='gray').encode(y='y')

    chart_architectures1 = alt.Chart(source.reset_index()).mark_circle(size=80, stroke='black').encode(
        x=alt.X('perfQ', axis=alt.Axis(title='performance'), scale=alt.Scale(domain=(perf_min, perf_max))),
        y=alt.Y('reliability', axis=alt.Axis(title='reliability'), scale=alt.Scale(domain=(rel_min, rel_max))),
        color=color_clusters, #'cluster:N',
        fillOpacity=alt.condition(click, alt.value(1.0), alt.value(0.1)),
        strokeWidth=alt.condition("datum.prototype", alt.value(4), alt.value(1)),
        tooltip=tooltip,
        order='prototype:O'
    ).add_params(click).interactive()

    chart_architectures2 = alt.Chart(source.reset_index()).mark_circle(size=80, stroke='black').encode(
        x=alt.X('pas', axis=alt.Axis(title='pas'), scale=alt.Scale(domain=(pas_min, pas_max))),
        y=alt.Y('#changes', axis=alt.Axis(title='#changes'), scale=alt.Scale(domain=(changes_min, changes_max))),
        color=color_clusters, #'cluster:N',
        fillOpacity=alt.condition(click, alt.value(1.0), alt.value(0.1)),
        strokeWidth=alt.condition("datum.prototype", alt.value(4), alt.value(1)),
        tooltip=tooltip,
        order='prototype:O'
    ).add_params(click).interactive()

    chart_architectures3 = alt.Chart(source.reset_index()).mark_circle(size=80, stroke='black').encode(
        x=alt.X('pas', axis=alt.Axis(title='pas'), scale=alt.Scale(domain=(pas_min, pas_max))),
        y=alt.Y('perfQ', axis=alt.Axis(title='perfQ'), scale=alt.Scale(domain=(perf_min, perf_max))),
        color=color_clusters, #'cluster:N',
        fillOpacity=alt.condition(click, alt.value(1.0), alt.value(0.1)),
        strokeWidth=alt.condition("datum.prototype", alt.value(4), alt.value(1)),
        tooltip=tooltip,
        order='prototype:O'
    ).add_params(click).interactive()

    hconcat = ((chart_architectures1 + perf_threshold_x + rel_threshold_y) | (chart_architectures2 + pas_threshold_x + changes_threshold_y) | (chart_architectures3 + pas_threshold_x + perf_threshold_y)| legend_clusters).configure_axis(
      labelFontSize=14,
      titleFontSize=14
    ).properties(title='Quality Attribute space: ' + title
    ).configure_title(fontSize=16, anchor='middle')#.configure_range(category={'scheme': CLUSTERS_COLORS})
  
    return hconcat


  # It requires the CLUSTERS and CLUSTERS_COLORS variables to be set!
  def show_scatter_plot(self, x, y, source=None, prototypes=[], title=None, show_all_labels=False, pfonly=False):

    tooltip = self.TOOLTIP
    if title is None:
      title = self.PROJECT_NAME

    if (source is None) and (not pfonly):
      source = self.tagged_objectives_df
    if (source is None) and (pfonly):
      source = self.get_pareto_front().merge(self.tagged_objectives_df, on=self.objectives)
    source = ArchitectureSpaceAnalyzer._add_cluster_prototypes(source, prototypes)

    click = alt.selection_point()
    selection_clusters = alt.selection_point(fields=['cluster'])
    #selection_prototypes = alt.selection_multi(fields=['prototype'])
    #input_dropdown = alt.binding_select(options=['True','False'], name='Reference points (only)')
    #selection_prototypes = alt.selection_single(fields=['prototype'], bind=input_dropdown)

    clabels = [self.CLUSTER_LABELS[idx] for idx in self.CLUSTERS]
    hexa_colors = self.CLUSTERS_COLORS
    if self.LABELS_COLORS is not None:
      hexa_colors = [self.LABELS_COLORS[lb] for lb in clabels]
    color_clusters = alt.condition(selection_clusters|click,
                      #alt.Color('cluster:N', legend=None, scale=alt.Scale(scheme='category10')),
                      alt.Color('cluster:O', legend=None, scale=alt.Scale(domain=list(self.CLUSTERS), range=hexa_colors)),
                      alt.value('transparent') #https://vega.github.io/vega/docs/schemes/#reference
                      )

    legend_clusters = alt.Chart(source).mark_point(size=150).encode(
        y=alt.Y('cluster:N', axis=alt.Axis(orient='right')),
        #y=alt.Y('cluster_label', axis=alt.Axis(orient='right')),
        color=color_clusters
    ).add_params(selection_clusters)

    extra = 0.2
    #changes_min = source[y].min()-extra
    #changes_max = source[y].max()+extra
    #pas_min = source[x].min()-extra
    #pas_max = source[x].max()+extra
    changes_min, changes_max = self._get_minmax(y)
    if (changes_min is None) and (changes_max is None):
        changes_min = source[y].min()
        changes_max = source[y].max()
    changes_min = changes_min-extra
    changes_max = changes_max+extra
    pas_min, pas_max =  self._get_minmax(x)
    if (pas_min is None) and (pas_max is None):
        pas_min = source[x].min()
        pas_max = source[x].max()
    pas_min = pas_min-extra
    pas_max = pas_max+extra

    pas_threshold_x = alt.Chart(pd.DataFrame({'x': [pas_min+extra]})).mark_rule(color='gray').encode(x='x')
    changes_threshold_y = alt.Chart(pd.DataFrame({'y': [changes_min+extra]})).mark_rule(color='gray').encode(y='y')

    if not show_all_labels:
      color_labels = color_clusters
    else:
      all_labels = list(self.LABELS_COLORS.keys())
      all_colors = list(self.LABELS_COLORS.values())
      #print(all_labels)
      #print(all_colors)
      color_labels = alt.Color('label', legend=None, scale=alt.Scale(domain=all_labels, range=all_colors))

    chart_architectures2 = alt.Chart(source.reset_index()).mark_circle(size=80, stroke='black').encode(
        x=alt.X(x, axis=alt.Axis(title=x), scale=alt.Scale(domain=(pas_min, pas_max))),
        y=alt.Y(y, axis=alt.Axis(title=y), scale=alt.Scale(domain=(changes_min, changes_max))),
        #color= color_clusters, #'cluster:N',
        color=color_labels,
        fillOpacity=alt.condition(click, alt.value(1.0), alt.value(0.1)),
        strokeWidth=alt.condition("datum.prototype", alt.value(4), alt.value(1)),
        tooltip=tooltip,
        order='prototype:O'
    ).add_params(click).interactive()

    hconcat = ((chart_architectures2 + pas_threshold_x + changes_threshold_y) | legend_clusters).configure_axis(
      labelFontSize=14,
      titleFontSize=14
    ).properties(title='Objective space: ' + title
    ).configure_title(fontSize=16, anchor='middle')#.configure_range(category={'scheme': CLUSTERS_COLORS})
  
    return hconcat


  def show_petal_plot(self, df=None, size=None, labels=False): # df can be an arbitrary set of points
    if df is None:
      df = self.centroids_df
    if (df.shape[0] == 1): # Only one rows
      scaled_values = df[self.objectives].values
      min = df[self.objectives].iloc[0].min()
      max = df[self.objectives].iloc[0].max()
    else:
      scaled_values = MinMaxScaler().fit_transform(df[self.objectives])
      min = 0
      max = 1

    if labels:
      #titles = [t+' ('+str(p)+'%) '+str(s) for t,p,s in zip(df.label, df.instances,df.solID)]
      if ('label' in df.columns) and ('instances' in df.columns):
        titles = [t+' ('+str(p)+'%)' for t,p in zip(df.label, df.instances)]
      else:
        titles = [t for t in df.label]
    else: 
      titles = []
      #titles = [t for t in centroids_df.label]
    plot = Petal(bounds=[min,max], title=titles, reverse=False, # Larger area means higher value?
             labels=list(self.objectives), figsize=size, tight_layout=True)
    plot = plot.add(scaled_values)
    return plot.show()


  def _make_radar_plot(self, prototypes, qa_goals, cluster_choice, normalize, title, size):
  
    categories = qa_goals
    categories = [*categories, categories[0]]
  
    names = []
    centroids = dict()
    colors = dict()
    #n_instances = prototypes['instances'].sum()
    for index, row in prototypes.iterrows():
      label = row['label']
      #percentage = round(100 * row['instances'] / n_instances, 2)
      #print("item"+str(index), list(row[categories].values), "name: ", label, percentage, '%')
      items = list(row[categories].values)
      items = [*items, items[0]]
      if (len(cluster_choice) == 0) or (index in cluster_choice):
        key = label #+' ('+str(percentage)+'%)'
        names.append(key)
        centroids[key] = items
      if self.LABELS_COLORS is None:
        colors[key] = self.CLUSTERS_COLORS[index]
      else:
        lb = self.CLUSTER_LABELS[index]
        colors[key] = self.LABELS_COLORS[lb]

    #print(colors)
    data = [go.Scatterpolar(r=centroids[x],theta=categories,name=x,line_color=colors[x],opacity=0.6, line_width=4) for x in names]
    #data = [go.Scatterpolar(r=centroids[x],theta=categories,fill='toself',name=x, fillcolor=colors[x],line_color=colors[x],opacity=0.6) for x in names]
    fig = go.Figure(data=data,
                    layout=go.Layout(title=go.layout.Title(text=title), width=size[0]*100, height=size[1]*100,
                                 polar={'radialaxis': {'visible': True}}, showlegend=True)
    )
    fig.show(renderer="colab")


  # Provide different visualizations for the cluster centroids (e.g., parallel plot, radar)
  def show_cluster_centroids(self, normalize=True, axis_choice=[], size=(10, 6), title=None):   
    if len(axis_choice) == 0:
      qa_goals = self.objectives
    else:
      qa_goals = axis_choice
    if len(qa_goals) < 2:
      print("Warning: at least 2 axis should be given -", qa_goals)

    # These should be the "real" centroids (not the theoretical ones)
    prototypes = self.centroids_df.reset_index(drop=True) #get_enriched_centroids(sample_df, normalize=normalize, ctype='r')
    prototypes = prototypes[qa_goals + ['cluster','label']]

    standardized = ''
    if normalize:
      standardized = ' (standardized)'
      scaler = StandardScaler() # Shouldn't be MinMaxScaler?
      scaled_features = scaler.fit_transform(prototypes[self.objectives].values)
    else:
      scaled_features = prototypes[self.objectives].values
    scaled_prototypes = pd.DataFrame(scaled_features, index=prototypes.index, columns=self.objectives)
    scaled_prototypes['cluster'] = prototypes['cluster']
    scaled_prototypes['label'] = prototypes['label']

    if title is None:
      title = self.PROJECT_NAME + ' - cluster centroids'
    title = title + standardized

    clusters = list(set(prototypes['cluster']))
    self._make_radar_plot(scaled_prototypes, qa_goals, cluster_choice=clusters, normalize=normalize, title=title, size=size)

    return prototypes

###########################################

  def get_pareto_front(self, objectives=None, maximize=False):
    if objectives is None:
      objectives = self.objectives

    matrix_df = self.objectives_df[objectives].copy()
    if 'perfQ' in objectives:
      matrix_df['perfQ'] = (-1)*matrix_df['perfQ']
    if 'reliability' in objectives:
      matrix_df['reliability'] = (-1)*matrix_df['reliability']

    # Compute Pareto front
    matrix = ArchitectureSpaceAnalyzer.pareto_efficient(matrix_df.values, maximize=maximize) 
    matrix_df = pd.DataFrame(matrix, columns=objectives).drop_duplicates().reset_index(drop=True)

    if 'perfQ' in objectives:
      matrix_df['perfQ'] = (-1)*matrix_df['perfQ']
    if 'reliability' in objectives:
      matrix_df['reliability'] = (-1)*matrix_df['reliability']

    # It always stores the un-normalized/un-inverted Pareto front
    self.pareto_front_df = matrix_df #.copy()
    return matrix_df


  def compute_pareto_front(self, objectives=None, normalize=True, invert_max=False, maximize=False):
    if objectives is None:
      objectives = self.objectives

    matrix_df = self.objectives_df[objectives].copy()
    if normalize:
      matrix = MinMaxScaler().fit_transform(matrix_df.values)
      matrix_df = pd.DataFrame(matrix, columns=objectives)    
    
    if 'perfQ' in objectives:
      matrix_df['perfQ'] = (-1)*matrix_df['perfQ']
    if 'reliability' in objectives:
      matrix_df['reliability'] = (-1)*matrix_df['reliability']

    # Compute Pareto front
    matrix = ArchitectureSpaceAnalyzer.pareto_efficient(matrix_df.values, maximize=maximize) 
    matrix_df = pd.DataFrame(matrix, columns=objectives).drop_duplicates().reset_index(drop=True)
   
    if 'perfQ' in objectives:
      matrix_df['perfQ'] = (-1)*matrix_df['perfQ']
    if 'reliability' in objectives:
      matrix_df['reliability'] = (-1)*matrix_df['reliability']
    
    # Invert if needed by some metric
    if invert_max and ('perfQ' in objectives):
      if normalize:
        matrix_df['perfQ'] = 1 - matrix_df['perfQ']
      else:
        matrix_df['perfQ'] = (-1)*matrix_df['perfQ']
    if invert_max and ('reliability' in objectives):
      if normalize:
        matrix_df['reliability'] = 1 - matrix_df['reliability']
      else:
        matrix_df['reliability'] = (-1)*matrix_df['reliability']

    return matrix_df

  @staticmethod
  def pareto_efficient(pts, maximize=False):
    'returns Pareto efficient row subset of pts'
    # sort points by decreasing sum of coordinates
    if maximize:
      pts = pts[pts.sum(1).argsort()[::-1]]
    else:
      pts = pts[pts.sum(1).argsort()]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    undominated = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        # process each point in turn
        n = pts.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        if not maximize:
          undominated[i+1:n] = (pts[i+1:] <= pts[i]).any(1) 
        else:
          undominated[i+1:n] = (pts[i+1:] >= pts[i]).any(1) 
        # keep points undominated so far
        pts = pts[undominated[:n]]
        undominated = np.array([True]*len(pts))

    return pts

  # Coverage indicator between two set (Pareto fronts). Assuming that all the objective values are minimized
  @staticmethod
  def compute_coverage(front_a, front_b):
    count_dominated_b = 0
    for b in front_b.values:
      weakly_domination = False
      for a in front_a.values:
        if all(a <= b):
          weakly_domination = True
      if weakly_domination:
        count_dominated_b = count_dominated_b + 1
    if front_b.shape[0] == 0:
      return math.inf
    return count_dominated_b / front_b.shape[0]


  def compute_gd(self, reference_front, objectives=None, normalize=True):      
    pf_df = self.compute_pareto_front(objectives=objectives, normalize=normalize, invert_max=True)
    
    if objectives is None:
      objectives = self.objectives
    ind = GD(reference_front[objectives].values)
    return ind(pf_df.values)
  

  def compute_hv(self, ref_point, objectives=None, normalize=True):
    pf_df = self.compute_pareto_front(objectives=objectives, normalize=normalize, invert_max=True)
    
    ind = HV(ref_point)
    return ind(pf_df.values)
  
  def compute_nps(self, objectives=None):
    if self.pareto_front_df is None:
      self.pareto_front_df = self.get_pareto_front(objectives=objectives)
    
    nps = self.pareto_front_df.shape[0] #len(set([tuple(x) for x in self.pareto_front_df.values]))
    return nps

  
  def compute_nsolutions(self, objectives=None):
    if objectives is None:
      objectives = self.objectives
    nsolutions = self.objectives_df[objectives].drop_duplicates().shape[0]
    return nsolutions

  def compute_nsequences(self):
    nsequences = self.refactions_df.drop_duplicates().shape[0]
    return nsequences


  def compute_igdplus(self, reference_front, objectives=None, normalize=True):      
    pf_df = self.compute_pareto_front(objectives=objectives, normalize=normalize, invert_max=True)
    
    if objectives is None:
      objectives = self.objectives
    ind = IGDPlus(reference_front[objectives].values)
    return ind(pf_df.values)
  

  def compute_all_metrics(self, reference_front=None, ref_point=None, objectives=None, normalize=True):
    my_front = self.compute_pareto_front(objectives=objectives, normalize=normalize, invert_max=True)
    
    dict_metrics = dict()
    dict_metrics['C_AB'] = ArchitectureSpaceAnalyzer.compute_coverage(my_front, reference_front)
    dict_metrics['C_BA'] = ArchitectureSpaceAnalyzer.compute_coverage(reference_front, my_front)
    #dict_metrics['GD'] = self.compute_gd(reference_front, objectives, normalize)
    dict_metrics['IGDPlus'] = self.compute_gd(reference_front, objectives, normalize)
    dict_metrics['HV'] = self.compute_hv(ref_point, objectives, normalize)
    # Remove it for now (it can be used as a separate metric)
    #dict_metrics['S'] = self.compute_spacing(reference_front, objectives, normalize)
    dict_metrics['NPS'] = self.compute_nps(objectives)
    dict_metrics['NSolutions'] = self.compute_nsolutions(objectives)
    dict_metrics['NSequences'] = self.compute_nsequences()
    dict_metrics['DE'] = self.compute_density_entropy(objectives, normalize)

    return dict_metrics

###########################################

  def create_encoder(self, operations):
    le = LabelEncoder()
    le.fit(list(operations))
    #print('Labels:', le.classes_)        
    if len(le.classes_) > len(self.ALPHABET):
      print("Warning: the number of classes to be encoded",len(le.classes_),"is larger than the available codes", len(self.ALPHABET))
  
    return le
    

  # Encode refactoring actions as sequences of letters of the alphabet
  def encode_operations(self, encoder=None, operations=None, ssplit='#', pfonly=False):
    if not pfonly:
      df = self.refactions_df
      print("encoding refactions:", df.shape[0])
    else:
      merged_df = self.get_pareto_front().merge(self.objectives_df, on=self.objectives)
      valid_solids = merged_df['solID'].unique()
      df = self.refactions_df[self.refactions_df['solID'].isin(valid_solids)]
      print("encoding refactions:", df.shape[0])

    if operations is None:
      operations = self.ALL_REFACTORINGS
    if encoder is None:
      encoder = self.create_encoder(operations=operations) 

    seq_ids = df['solID'].values.tolist()
    #print(seq_ids)
    encoded_df = df.copy().drop('solID', axis=1)
    for c in encoded_df.columns:
      encoded_df[c] = encoder.transform(df[c])

    codes = [] # Encoding with letters of the alphabet (a finite number of letters)
    for lst, seq in zip(encoded_df.values.tolist(), seq_ids):
      if self.use_alphabet:
        sub = ''.join([self.ALPHABET[x] for x in lst])
      else:
        sub = ssplit.join([str(x) for x in lst])
      #print(lst,sub)
      codes.append((sub,seq)) # pair <sub,id>
    return codes, encoder


  # Decode sequences of letters of the alphabet in terms of (original) refactoring actions
  def decode_operations(self, codes, encoder, ssplit='#'):
    result = dict()
    for c in codes: # Decoding the refactoring actions
      if self.use_alphabet:
        indices = [self.ALPHABET.index(v) for v in list(c)]
      else:
        lst = c.split(ssplit)
        indices = [int(v) for v in lst]
      result[c] = list(encoder.inverse_transform(indices))
    return result
  
  
  def get_candidate_operations(self, op_columns=None, position=0, pfonly=False):
    if not pfonly:
      df = self.refactions_df
      print("refactions:", df.shape[0])
    else:
      merged_df = self.get_pareto_front().merge(self.objectives_df, on=self.objectives)
      valid_solids = merged_df['solID'].unique()
      df = self.refactions_df[self.refactions_df['solID'].isin(valid_solids)]
      print("refactions:", df.shape[0])

    if (op_columns is None) and (len(df.columns) == 3):
      op_columns = ['op1','op2']
    if (op_columns is None) and (len(df.columns) == 5):
      op_columns = ['op1','op2','op3','op4']
    
    if (position is None) or (position == 0):
      all_ops = df[op_columns].T.to_numpy().flatten()
    else:
      all_ops = df[op_columns[position-1]].T.to_numpy().flatten()
    print("all operations:", len(all_ops))
    return all_ops # This is a list 


  def _get_sequence_count(self, solID):
    op_columns = None
    if (self.refactions_df.shape[1] == 3):
      op_columns = ['op1','op2']
    elif (self.refactions_df.shape[1] == 5):
      op_columns = ['op1','op2','op3','op4']
    all_sequences = [tuple(s) for s in self.refactions_df[op_columns].values]
    sequence_pattern = tuple(self.refactions_df[self.refactions_df.solID == solID][op_columns].values[0])
    return all_sequences.count(sequence_pattern)


  def get_sequences_distribution(self, k=None, size=(10,5), normalize=False, show_chart=True):
    op_columns = None
    if (self.refactions_df.shape[1] == 3):
      op_columns = ['op1','op2']
    elif (self.refactions_df.shape[1] == 5):
      op_columns = ['op1','op2','op3','op4']
    all_sequences = [tuple(s) for s in self.refactions_df[op_columns].values]
    sequences_counter = Counter(all_sequences)

    if show_chart:
      top_k_sequences = OrderedDict(sequences_counter.most_common(k)) 
      top_k_sequences = dict(reversed(list(top_k_sequences.items()))) # Reverse order?
      unique_seqs = [str(x) for x in top_k_sequences.keys()]
      count_seqs = list(top_k_sequences.values())
      #print(type(unique_seqs), unique_seqs)
      #print(type(count_seqs), count_seqs)
      if normalize:
        n = len(all_sequences)
        count_seqs = [s/n for s in count_seqs]
      
      fig = plt.figure(figsize=size)
      plt.barh(unique_seqs, count_seqs)
      plt.title('Frequency of sequences'+' ('+self.PROJECT_NAME+')')
      if normalize:
        plt.xlim([0.0,1.0])
      plt.show()

    return sequences_counter


  def show_refactions_distribution(self, k=None, size=(20,10), option=None, normalize=False, pfonly=False):
    
    if (option is None) or (option > 0):
      all_ops = self.get_candidate_operations(op_columns=None, position=option, pfonly=pfonly)
      unique, counts = np.unique(list(all_ops), return_counts=True)
      if k is not None:
        idx = (-counts).argsort()[:k] # Indices of the top-5 highest values
        unique = unique[idx]
        counts = counts[idx]
      if normalize and (len(all_ops) > 0):
        counts = counts / len(all_ops)

      fig = plt.figure(figsize=size)
      plt.barh(unique, counts)
      #plt.xticks(rotation=60, ha='right')
      plt.title('Frequency of refactoring actions - '+str(option)+' ('+self.PROJECT_NAME+')')
      if normalize:
        plt.xlim([0.0,1.0])
      plt.show()
      return unique, counts
    
    else: # Option is 0
      nplots = self.refactions_df.shape[1]
      fig, axs = plt.subplots(nplots-1, figsize=size, sharex=True)
      unique_list = []
      counts_list = []
      color_list = ['red','green','cyan','brown']
      for n in range(1, nplots):
        #print("plot:",n)
        all_ops = self.get_candidate_operations(op_columns=None, position=n, pfonly=pfonly)
        unique, counts = np.unique(list(all_ops), return_counts=True)
        if k is not None:
          idx = (-counts).argsort()[:k] # Indices of the top-5 highest values
          unique = unique[idx]
          counts = counts[idx]
        if normalize and (len(all_ops) > 0):
          counts = counts / len(all_ops)
        unique_list.append(unique)
        counts_list.append(counts)

        axs[n-1].barh(unique, counts, color=color_list[n-1])
        #plt.xticks(rotation=60, ha='right')
        axs[n-1].set_title('Frequency of refactoring actions - '+str(n)+' ('+self.PROJECT_NAME+')')
      
      if normalize:
        plt.xlim([0.0,1.0])
      plt.show()
      return unique_list, counts_list


  def get_codes(self, encoder=None, ops=None, use_alphabet=True, pfonly=False):
    self.use_alphabet = use_alphabet
    candidate_ops = set(self.get_candidate_operations(ops, pfonly=pfonly))
    print(len(candidate_ops), "Distinct (individual) refactoring actions:", candidate_ops)
  
    # Codification of sequences of refactoring actions
    # codes is a list of tuples <sub,id>
    codes, encoder = self.encode_operations(encoder=encoder, operations=candidate_ops, pfonly=pfonly)
  
    print("Sequence codes:", len(codes), codes)
    seq_codes = [c[0] for c in codes]
    unique_codes = list(set(seq_codes))
    print("---unique:", len(unique_codes), unique_codes)
    duplicate_codes = set([x for x in seq_codes if seq_codes.count(x) > 1])
    print("---duplicates:", len(duplicate_codes), duplicate_codes)

    #unique, counts = np.unique(list(candidate_ops), return_counts=True)
    #plt.bar(unique, counts)
    #plt.xticks(rotation=60, ha='right')
    #plt.title('Frequency of refactoring actions')
    #plt.show()

    #print(encoder.classes_)
    return codes, encoder


  def add_sequence_prefix(self, prefix): # TODO: Fix this hack later!
    df_expanded = self.refactions_df.copy()
    if len(df_expanded.columns) == 3:
      df_expanded.columns = ['solID', 'op3', 'op4']
    df_expanded['op1'] = prefix[0]
    df_expanded['op2'] = prefix[1]

    self.refactions_df = df_expanded[['solID', 'op1', 'op2', 'op3', 'op4']]
    return self.refactions_df


  # Other similarity metrics are also possible (e.g., hamming or jaccard)
  def _similarity(self, a, b, metric='hamming'):
    if metric == 'smatcher':
      matcher = SequenceMatcher(None, a, b)
      #print(matcher.get_matching_blocks())
      return matcher.ratio()
    
    if metric == 'damerauLevenshtein':
      if self.use_alphabet:
        return damerauLevenshtein([*a],[*b], similarity=True) 
      else: # This is the only working metric (so far) when use_alphabet=False
        return damerauLevenshtein(a.split('#'),b.split('#'), similarity=True) 
    
    if metric == 'hamming':
      if self.use_alphabet:
        return 1-distance.hamming([*a],[*b])
      else: # This is the only working metric (so far) when use_alphabet=False
        return 1-distance.hamming(a.split('#'),b.split('#'))
    
    if metric == 'jaccard':
      a1 = [self.ALPHABET.index(x) for x in a]
      b1 = [self.ALPHABET.index(x) for x in b]
      return distance.jaccard(a1,b1)

    return None

###########################################

  # Compute a string similarity between the encoded sequences of refactoring actions
  def compute_sequence_distances(self, sequences, metric='hamming', non_zero=0.0):
    #print(metric)
    distance_dict = dict()
    for n in sequences:
      distance_dict[n] = dict()
      for m in sequences:
        #distance_dict[n][m] = similarity(m,n) + non_zero
        distance_dict[n][m] = (1 - self._similarity(m,n,metric))
        if distance_dict[n][m] > 1:
          distance_dict[n][m] = 1.0
        if distance_dict[n][m] <= 0:
          distance_dict[n][m] = 0.0 + non_zero
  
    return distance_dict


  @staticmethod
  def compute_tree_distances(trie, normalize=None, non_zero=0.0, ssplit=''):
      final_nodes = [n for n in trie.nodes() if trie.out_degree(n) == 0]
      id_sequences = [nx.shortest_path(trie,0, n) for n in final_nodes]
      print(len(id_sequences), "sequences")
      
      labels = dict()
      for lst in id_sequences:
          #lst.remove(0)
          k = lst[-1]
          lst = [trie.nodes[x]['source'] for x in lst if (x!=0)]
          labels[k] = ssplit.join(lst)
      
      n_nodes = 1
      if normalize is not None:
          n_nodes = normalize #trie.number_of_nodes()-1
      undirected = trie.to_undirected()
      distances_dict = dict()
      for n in final_nodes:
          dist_n = dict()
          for m in final_nodes:
              if nx.has_path(undirected, n,m) or nx.has_path(undirected, m,n):
                  d = nx.shortest_path_length(undirected, source=n, target=m) / n_nodes
                  if d <= 0:
                      d = non_zero
                  dist_n[labels[m]] = d
          distances_dict[labels[n]] = dist_n
      return distances_dict
  

  # Convert the sequences into a graph, in which similar sequences get closer in the graph layout
  # Using Kamada Kawai layout: the closer the distance between nodes, the closer their positions in the graph layout
  def get_sequence_graph(self, sequences_dict, show=True, encoder=None, size=(5,5), info=None, 
                              title=None, metric='hamming', margin=0.0, distances=None):
    if title is None:
      title = self.PROJECT_NAME
    graph = nx.DiGraph()
    for s in sequences_dict.keys():
      seq_id = sequences_dict[s]
      c = -1
      lb = '- / - / - / -'
      if info is not None:
        row = self.tagged_objectives_df[self.tagged_objectives_df.solID == seq_id]
        c = row['cluster'].values[0]
        lb = row['label'].values[0]
      narch = self._get_sequence_count(seq_id)
      #print(seq_id,narch)
      graph.add_node(s, seq=seq_id, cluster=c, label=lb, ninstances=narch)
  
    if show:
      fig = plt.figure(figsize=size)

      #distances = self._compute_sequence_distances([*sequences_dict]) #, non_zero=0.1)
      print(metric)
      if distances is None:
          distances = self.compute_sequence_distances([*sequences_dict], metric=metric, non_zero=0.1)
      pos = nx.kamada_kawai_layout(graph, dist=distances)

      labels_dict = nx.get_node_attributes(graph, 'seq')
      new_labels_dict = { k:k+'_'+str(v) for (k,v) in labels_dict.items() }

      if (info is not None) and (self.CLUSTERS_PALETTE is not None):
        cluster_dict = nx.get_node_attributes(graph, 'cluster')
        colors = [self.CLUSTERS_PALETTE[v] for (k,v) in cluster_dict.items()]
      else:
        colors = ['lightgray']*graph.number_of_nodes()
      #print(colors)

      sizes_dict = nx.get_node_attributes(graph, 'ninstances')
      #print(sizes_dict)
      sizes = [300+(3000*s/self.refactions_df.shape[0]) for (k,s) in sizes_dict.items()]

      #nx.draw(graph, pos, with_labels=True, node_color='cyan', node_size=1000)
      nodes = nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes)
      labels = nx.draw_networkx_labels(graph, pos, labels=new_labels_dict)
      nodes.set_edgecolor('darkgray')
      #nx.draw(graph, pos, with_labels=True, node_color='cyan', node_size=1000, labels=labels)

      ax = plt.gca()
      ax.set_title(title)
      #fig.suptitle(title, fontsize=15)

      plt.margins(margin)
      fig.tight_layout() 
      plt.grid(visible=False)
      plt.box(False)
      
      plt.show()
  
    return graph


  # Check if a given path exists in the trie
  @staticmethod
  def _find_node(trie, path):
    root = trie.nodes[0]
    #print("tree root:", root)

    n = 0
    for v in path:
    
      matching = False
      for s in trie.successors(n):
        #print(tries.nodes[s])
        if trie.nodes[s]['source'] == v:
          matching = True
          #print(v, "...got a match=", s)
          n = s
          break
    
      if not matching:
        #print("matching aborted for", v)
        n = None
        break
  
    return n


  def _get_labels(self, labels, le, trim=False):
    original_labels = dict()
    for k,v in labels.items():
      if v is not None:
        ix = self.ALPHABET.index(v) # Get the index of the letter 
        #print(k,v,ix)
        p = le.classes_[ix].find('(')
        if trim and (p != -1):
          original_labels[k] = le.classes_[ix][0:p]
        else:
          original_labels[k] = le.classes_[ix]
      else: 
        original_labels[k] = v

    return original_labels


  def _get_node_as_sequence(trie, x, sjoin=''):
    path = nx.shortest_path(trie, source=0, target=x)
    seq = [trie.nodes[n]['source'] for n in path]
    #print("path for:", x, path, seq)
    seq.pop(0)
    seq = sjoin.join(seq)
    return seq 


  def _build_prefix_tree(sequences, ssplit=None):
    all_paths = []
    for s in sequences:
      if ssplit is None:
        seq = list(s)
      else:
        seq = s.split(ssplit)
      all_paths.append(seq)
    #print(all_paths)

    trie = nx.DiGraph()
    trie.add_node(0, source=None)

    count = 0
    prefix_dict = dict()
    for p in all_paths:
      for n in range(1,len(p)+1):
        subseq = p[0:n]
        prefix = '.'.join(subseq)
        #print(p, prefix)
        if prefix not in prefix_dict.keys():
          count += 1
          prefix_dict[prefix] = count
          trie.add_node(count, source=subseq[-1], prefix=prefix)
      #node_dict[n] = int(n)
    #print(count, "nodes added")

    for p in all_paths:
      for n in range(1,len(p)+1):
        subseq = p[0:n]
        prefix = '.'.join(subseq)
        node_subseq = prefix_dict[prefix]
        if (len(subseq) == 1) and (not trie.has_edge(0,node_subseq)):
            trie.add_edge(0,node_subseq)
        if len(subseq) > 1:
          prefix_prev_node = '.'.join(subseq[0:-1])
          #print(prefix_prev_node, '-->', prefix)
          prev_node_subseq = prefix_dict[prefix_prev_node]
          if not trie.has_edge(prev_node_subseq, node_subseq):
            trie.add_edge(prev_node_subseq, node_subseq)
    
    return trie


  # Convert the sequence of refactoring actions into a prefix tree (trie)
  def get_prefix_tree(self, sequences_dict, show=True, encoder=None, size=(5,5), layout='graphviz', prog='dot', title=None, info=None, margin=0.0):
    if title is None:
      title = self.PROJECT_NAME
    if self.use_alphabet:
      #trie = build_prefix_tree([*sequences_dict])
      trie = nx.prefix_tree([*sequences_dict]) # Create a prefix tree with the codes
      #print(trie.edges(data=True))
      #print(trie.nodes(data=True))
      trie.remove_node(-1) # The 'void' node (leaf), which should be removed
    else:
      trie = ArchitectureSpaceAnalyzer._build_prefix_tree([*sequences_dict], ssplit='#')
    
    # Default attributes for the trie
    nx.set_node_attributes(trie, 1.5, 'width')
    nx.set_node_attributes(trie, 'gray', 'color')
    nx.set_edge_attributes(trie, 1.5, 'width')  
    nx.set_edge_attributes(trie, 'gray', 'fillcolor')  

    # Get the labels for the nodes from their attributes
    labels = nx.get_node_attributes(trie, 'source') 
    if encoder is not None:
      #labels = get_labels(labels, encoder)
      labels = self._get_labels(labels, encoder, trim=True)
      nx.set_node_attributes(trie, None, "refaction") 
      i = 0 # Update the refactoring action for each node
      for x, data in trie.nodes(data=True):
        data['refaction'] = labels[i]
        i += 1
    #print(labels)

    final_nodes = set()
    for s in sequences_dict.keys():
      if self.use_alphabet:
        n = ArchitectureSpaceAnalyzer._find_node(trie, s)
      else:
        n = ArchitectureSpaceAnalyzer._find_node(trie, s.split('#'))
      #print("Finding",s,n)
      if n is not None:
        final_nodes.add(n)
    #final_nodes = sorted(final_nodes)
    #print(len(final_nodes), final_nodes)

    #nx.set_node_attributes(trie, self.CLUSTERS_COLORS[-1], "color")
    nx.set_node_attributes(trie, '#FFFFFF', "fillcolor") # white
    nx.set_node_attributes(trie, '- / - / - / -', "label") 
    nx.set_node_attributes(trie, 0, "ninstances") 
    assigned_colors = set()
    for x, data in trie.nodes(data=True):
      if x in final_nodes:
        data['color'] = "black" 
        if self.use_alphabet:
          seq = ArchitectureSpaceAnalyzer._get_node_as_sequence(trie, x)
        else:
          seq = ArchitectureSpaceAnalyzer._get_node_as_sequence(trie, x, sjoin='#')
        #print("final node:", seq, x)
        
        seq_id = sequences_dict[seq]
        data['fillcolor'] = 'lightgray'
        if info is not None:
          row = self.tagged_objectives_df[self.tagged_objectives_df.solID == seq_id]
          c = row['cluster'].values[0]
          lb = row['label'].values[0]
          if self.CLUSTERS_PALETTE is not None:
            data['fillcolor'] = self.CLUSTERS_PALETTE[c]
          data['label'] = lb
        data['ninstances'] = self._get_sequence_count(seq_id)
        assigned_colors.add(data['fillcolor'])

    #line_colors = ['black' if (x in final_nodes) else 'gray' for x in trie.nodes()]
    line_colors = nx.get_node_attributes(trie, 'color').values()
    node_colors = nx.get_node_attributes(trie, 'fillcolor').values()
    sizes_dict = nx.get_node_attributes(trie, 'ninstances')
    sizes = [300+(3000*s/self.refactions_df.shape[0]) if (k in final_nodes) else 1000 for (k,s) in sizes_dict.items()]

    linewidths = len(trie.nodes())*[1.5]
    if info is not None:
      #cluster_ids = CLUSTERS_COLORS[0:len(CLUSTERS)]
      #print(trie.nodes())
      #print(assigned_colors)
      cluster_ids = assigned_colors
      #entropies = get_nodes_entropy(trie, cluster_ids)
      #print(entropies)
      #eps = 0.7
      #linewidths = [6.0 if data['entropy'] >= eps else 1.5 for x, data in trie.nodes(data=True)] # Nodes with high entropy mean high variability in the outcomes
      #print("Key nodes (high entropy): ", [(x,data['entropy']) for x,data in trie.nodes(data=True) if (data['entropy'] >= eps)])
      #eps = 0.4
      #linewidths = [6.0 if (data['entropy'] <= eps) and (data['depth'] <= 2) else 1.5 for x,data in trie.nodes(data=True)] # Nodes with low variability mean that go to the same cluster
      #print("Key nodes (low entropy): ", [(x,data['entropy']) for x,data in trie.nodes(data=True) if (data['entropy'] <= eps)])

    if not show:
      return trie

    fig = plt.figure(figsize=size) 
    fig.suptitle(title, fontsize=20)
    #ax = plt.gca()  
    #ax.set_title(title)
    #plt.title(title)

    if layout == 'graphviz':
      #pos = graphviz_layout(trie, prog="dot", root=0)
      #pos = graphviz_layout(trie, prog="twopi", root=0)
      pos = graphviz_layout(trie, prog=prog, root=0)
      nodes = nx.draw_networkx_nodes(trie, pos, node_color=node_colors, node_size=sizes, linewidths=linewidths)
      #nodes.set_edgecolors('gray') 
      nodes.set_edgecolors(line_colors)
      labels = nx.draw_networkx_labels(trie, pos, labels=labels)
      edges = nx.draw_networkx_edges(trie, pos)
      #nx.draw(trie, pos, with_labels=True, node_color='cyan', node_size=1000, labels=labels)
    else:
      nodes = nx.draw_networkx_nodes(trie, pos=nx.spring_layout(trie), node_color='cyan', node_size=sizes)
      labels = nx.draw_networkx_labels(trie, pos=nx.spring_layout(trie))
      edges = nx.draw_networkx_edges(trie, pos=nx.spring_layout(trie))
      #nx.draw_spring(trie, with_labels=True, node_color='cyan', node_size=1000, labels=labels)

    plt.margins(margin)
    fig.tight_layout() 
    plt.grid(visible=False)
    plt.box(False)
    
    plt.show()
  
    return trie


###########################################

  # Convert trie to tree (for ZSS distance)
  @staticmethod
  def _get_zss_tree(trie, root=None):

    temp = nx.relabel_nodes(trie, {x: str(x) for x in trie.nodes()}) # Convert the node labels to strings
    if root is None:
      root = [n for (n, d) in temp.in_degree() if d == 0][0]
  
    print("root:", root, type(root))
    tree = nx.dfs_tree(temp, source=root)

    nodes_dict = {}
    for edge in tree.edges():
      if edge[0] not in nodes_dict:
        nodes_dict[edge[0]] = zss.Node(edge[0])
      if edge[1] not in nodes_dict:
        nodes_dict[edge[1]] = zss.Node(edge[1])
      nodes_dict[edge[0]].addkid(nodes_dict[edge[1]])

    return nodes_dict, root

  # Computation of edit distance (ZSS) between a pair of prefix trees (tries)
  @staticmethod
  def compute_zss_distance_for_tries(trie1, trie2=None, root1=None, root2=None):
    tree1, root1 = ArchitectureSpaceAnalyzer._get_zss_tree(trie1, root1)
    if trie2 is None:
      return zss.simple_distance(tree1[root1], tree1[root1])
    else:
      tree2, root2 = ArchitectureSpaceAnalyzer._get_zss_tree(trie2, root2)
      return zss.simple_distance(tree1[root1], tree2[root2])

###########################################

  def clusters_info(self):
    count_series = self.tagged_objectives_df['cluster'].value_counts()
    percentage_series = self.tagged_objectives_df['cluster'].value_counts(normalize=True)
    clusters = list(count_series.index)
    for k,c,p in zip(clusters, count_series.values, percentage_series.values):
      print("  cluster", k,":",c,"items","{:.2f}".format(round(100*p, 2)),"%", self.CLUSTER_LABELS[k])


  def _get_labels_as_tuples(self, labels, objectives=None):
    my_labels_set= set()
    for lb in labels:
      lb_tuple = tuple([x.strip() for x in lb.split('/')])
      if objectives is None:
        my_labels_set.add(lb_tuple)
      else:
        reduced_tuple = tuple([lb_tuple[i] for i in objectives])
        my_labels_set.add(reduced_tuple)
    return my_labels_set

  def intersect_cluster_labels(self, labels, objectives=None):
    my_labels_set= self._get_labels_as_tuples(self.CLUSTER_LABELS.values(), objectives)  
    another_set = self._get_labels_as_tuples(labels, objectives)

    return my_labels_set.intersection(another_set)


  def match_cluster_labels(self, labels, objectives=None):
    my_labels_set= self._get_labels_as_tuples(self.CLUSTER_LABELS.values(), objectives)  
    another_set = self._get_labels_as_tuples(labels, objectives)

    best_match = None
    for x in my_labels_set:
      best_score = 0
      best_match = None
      for y in another_set:
        h = distance.hamming(x,y)
        if h > best_score:
          best_score = h
          best_match = y 
      print(x, '-->', best_match, 'matching=', best_score)

  def distance_cluster_labels(self, labels, objectives=None):
    my_clabels = asa._get_labels_as_tuples(asa.CLUSTER_LABELS.values(), objectives)
    other_clabels = asa._get_labels_as_tuples(labels, objectives)
    x = [list(c) for c in my_clabels]
    y = [list(c) for c in other_clabels]
    matrix = squareform(pdist(x, y, lambda u, v: distance.hamming(u,v)))
      
    return pd.DataFrame(matrix, index=my_clabels, columns=other_clabels)
  
  
  # Utility function
  def plot_barchart_group(self, count_ops=None, n=4, width=0.2, colors=['r', 'g', 'b','orange'], 
                          ax=None, title='', ops=['op1','op2','op3','op4'], pfonly=False):
    
    if count_ops is None:
      unique_list = []
      counts_list = []
      n = self.refactions_df.shape[1]-1
      for i in range(1, n+1):
        #print("plot:", i)
        all_ops = self.get_candidate_operations(op_columns=None, position=i, pfonly=pfonly)
        unique, counts = np.unique(list(all_ops), return_counts=True)
        counts = counts / len(all_ops) # Normalization
        unique_list.append(unique)
        counts_list.append(counts)
      count_ops = (unique_list, counts_list)

    ind = np.arange(n) 
    candidate_ops = count_ops[0]
    op_values = count_ops[1]
    i = 0
    all_bars = []
    for op,c in zip(candidate_ops,colors):
      vals = [item[i] if (i < len(item)) else 0.0 for item in op_values]
      #print(vals)
      if ax is None:
        bar = plt.bar(ind+width*i, vals, width, color=c)
      else:
        bar = ax.bar(ind+width*i, vals, width, color=c)
      all_bars.append(bar)
      i += 1
  
    if ax is None:
      plt.xlabel("Sequence type")
      plt.ylabel('Percentage')
      plt.title("Distribution of refactoring actions per type "+title)
      plt.xticks(ind+width, ops)
      plt.legend(tuple(all_bars), tuple(count_ops[0][0]), bbox_to_anchor=(1.04, 1), loc="upper left")
      plt.ylim((0,1))
      plt.show()
    else:
      ax.set_xticks(ind+width) 
      ax.set_xticklabels(ops)
      ax.title.set_text(title)
  
    return tuple(all_bars), tuple(count_ops[0][0])


###########################################