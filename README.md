# Masterthesis
This repo provides code used for the thesis and datasets for additive and multiplicative noise model. Furthermore 3 videos of different regimes of the nullclines in 3D space generated with MAPLE are provided.

Data:\
The data for the 2D Model was generated using the Code "2D Model Grid Simulation.ipynb".\
We use a timeseries of 6000s and dt = 0.001s to simulate the 2D Model. \
We store the statistics of Up and Down states for each gridpoint in a matrix. \
Default parameters used:\
tau = 0.01s\
tau_a = 1s\
theta = 10mV\
r_max = 10Hz\
b = 0.4 mVs\
g = 0.25 1/mV \
N = 600

In the Data folder one can chose between additive and multiplicative noise data. Furthermore one can chose between two different timescales of the adaptation tau_a.\
Multiplicative Noise:\
Data for different population sizes N is provided. Furthermore for N = 600 data using gaussian noise and poisson noise is provided.\
The multiplicative noise model has noise in the excitation and adaptation.\
The file kramer_matrix.npy provided for each gridsimulation gives the estimated Kramers-Moyal coefficient without w.\
The file sigma_mult_w_matrix.npy provided for each gridsimulation gives the estimated Kramers-Moyal coefficient with w.

Additive Noise:
Here statistics obtained on the grid using Kramers-Moyal (noise is changing for each gridpoint) and constant noise over the whole grid is provided.\
Using sigma_h of Kramers-Moyal is done for N = 600 of the mult noise model.\
For Kramers-Moyal matched additive noise data with noise in excitation only and noise in excitation and adaptation is provided.

For constant noise over the grid we provide data of different noise strengths. These simulations only had noise in the excitation not in the adaptation.
