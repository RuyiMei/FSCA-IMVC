FSCA/IMVC

Introduction

This repository contains the MATLAB implementation of Bridging Feature and Structure: Consensus Anchor-Guided Bipartite Graph Learning for Incomplete Multi-View Clustering (FSCA/IMVC).

The proposed method integrates feature representation learning and bipartite graph structure learning through a cross-level self-supervised framework. Consensus anchors are employed to connect the feature and graph structure spaces, while predefined and adaptive bipartite graphs are jointly considered to support consensus graph learning under incomplete multi-view settings.

Project Structure

The project contains the following files and folders:

run.m: The main script for running experiments, configuring parameters, evaluating clustering performance, and saving results.

FSCA.m: The main implementation of the proposed FSCA/IMVC algorithm.

WebKB_2views/: Contains the WebKB two-view dataset used in the experiments.

Missing_data_generation/: Contains code for generating incomplete multi-view data and missing-view indicators.

measure/: Contains functions for evaluating clustering performance.

functions/: Contains auxiliary functions required by the algorithm.

results.xlsx: Stores experimental results and the corresponding parameter settings.

Requirements

The code is implemented in MATLAB and can be run using MATLAB R2023b.

Before running the experiments, ensure that the required datasets and MATLAB functions are available in the corresponding directories.

Usage

First, download the project and open MATLAB. Set the current working directory to the root folder containing run.m.

The main script automatically adds the project directories to the MATLAB search path.

To run the experiments, execute:

run('run.m');

The dataset used in the experiments can be selected by modifying the following line in run.m:

ds = {'WebKB_2views'};

To conduct experiments on other datasets, place the corresponding data files in the project directory and modify the dataset name accordingly.

Before running the clustering algorithm, prepare the incomplete multi-view data using the code in Missing_data_generation/. The missing-data construction process should ensure that each sample retains at least one observed view.

The experimental settings, including the missing rate, number of anchors, and model parameters, can be adjusted in run.m.

Experimental Results

After execution, the experimental results are saved in results.xlsx.

The recorded information includes the missing rate, model parameters, clustering evaluation metrics, number of iterations, and execution time.

The clustering evaluation metrics include ACC, NMI, Purity, F-score.
The workbook also contains a worksheet named Best_ACC, which is used to record the parameter configurations and corresponding results associated with the best clustering accuracy.

Note: The current run.m script deletes the existing results.xlsx file before starting a new experiment. Please back up previous results if necessary.
