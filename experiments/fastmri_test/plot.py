import os, sys, io, pathlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import pdb

class CPU_Unpickler(pkl.Unpickler):
  def find_class(self, module, name):
    if module == 'torch.storage' and name == '_load_from_bytes':
      return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
    else:
      return super().find_class(module, name)

def plot_spearman(methodnames,results_list):
  plt.figure(figsize=(8,1.5))
  sns.set_palette('pastel')
  # Crop sizes to 99%
  spearmans = [results['spearman'] for results in results_list]
  #df = pd.DataFrame({'Spearman Rank Correlation' : [results['spearman'] for results in results_list], 'Method': [method.replace(' ','\n') for method in methodnames]})
  #g = sns.scatterplot(data=df, x='Method', y='Spearman Rank Correlation', kind='bar')
  for j in range(len(methodnames)):
    plt.scatter(x=spearmans[j], y=[0,], s=70, label=methodnames[j])
  sns.despine(top=True, bottom=True, right=True, left=True)
  plt.tight_layout()
  plt.gca().set_yticks([])
  plt.gca().set_yticklabels([])
  plt.ylim([-0.1,1])
  plt.legend()
  plt.gca().tick_params(axis=u'both', which=u'both',length=0)
  plt.xlabel("Spearman rank correlation between heuristic and true residual")
  plt.savefig('outputs/fastmri-spearman.pdf')

def plot_size_violins(methodnames,results_list):
  plt.figure()
  sns.set_palette('pastel')
  # Crop sizes to 99%
  for results in results_list:
    results['sizes'] = torch.clamp(results['sizes'], min=0, max=2)
  df = pd.DataFrame({'Interval Size' : torch.cat([results['sizes'] for results in results_list]).tolist(), 'Method': [method.replace(' ','\n') for method in methodnames for i in range(results_list[0]['sizes'].shape[0])]})
  g = sns.violinplot(data=df, x='Method', y='Interval Size', cut=0)
  sns.despine(top=True, right=True)
  plt.yticks([0,1,2])
  plt.gca().set_yticklabels(['0%','50%','100%'])
  plt.tight_layout()
  plt.savefig('outputs/fastmri-sizes.pdf')

def plot_ssr(methodnames,results_list):
  plt.figure()
  sns.set_palette(sns.light_palette("salmon"))
  df = pd.DataFrame({'Difficulty': len(results_list)*['Easy', 'Easy-Medium', 'Medium-Hard', 'Hard'], 'Risk' : torch.cat([results['size-stratified risk'] for results in results_list]).tolist(), 'Method': [method.replace(' ','\n') for method in methodnames for i in range(results_list[0]['size-stratified risk'].shape[0])]})
  g = sns.catplot(data=df, kind='bar', x='Method', y='Risk', hue='Difficulty',legend=False)
  sns.despine(top=True, right=True)
  plt.legend(loc='upper right')
  plt.xlabel('')
  plt.tight_layout()
  plt.savefig('outputs/fastmri-size-stratified-risk.pdf')

def generate_plots():
  methodnames = ['Gaussian','Residual Magnitude','Quantile Regression']
  filenames = ['outputs/raw/results_fastmri_gaussian_78_0.0001_standard_standard.pkl','outputs/raw/results_fastmri_residual_magnitude_78_0.0001_standard_standard.pkl','outputs/raw/results_fastmri_quantiles_78_0.0001_standard_standard.pkl']
  # Load results
  results_list = []
  for filename in filenames:
    with open(filename, 'rb') as handle:
      results_list = results_list + [CPU_Unpickler(handle).load(),]
  # Plot spearman correlations
  plot_spearman(methodnames,results_list)
  # Plot size-stratified risks 
  plot_ssr(methodnames,results_list)
  # Plot size distribution
  plot_size_violins(methodnames,results_list)

if __name__ == "__main__":
  generate_plots()
