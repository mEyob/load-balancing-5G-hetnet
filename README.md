## Introduction

This repo provides a python simulation package for energy-aware load balancing of 
user traffic in 5G hetrogeneous networks.  As depicted in the
following figure, the network consists
of a single Macro cell and a number of small cells. User traffic is assumed to be an 
elastic data traffic following a Poisson arrival process. Small cells provide higher
data rate to users within their coverage area, but they can also be turned off 
to save energy when demand is low. Thus the task is to develop a dynamic load (un)balancing 
policy that creates energy saving opportunities by offloading user traffic to macro 
cell whenever possible. More specifically, the target is to minimize 
energy consumption under a given performance constraint (mean response time).
To this end, user request scheduling 
is modeled as a Constrained Markov Decision Process (CMDP).

<img src="hetnet-model.png" alt="perHr" style="width: 100px; height: 100px" />

# Converting constrained optimization to unconstrained
The constrained optimization 

<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{\Pi}{\text{minimize&space;}}&space;E[P^\Pi]&space;\\&space;\text{subject&space;to&space;}&space;E[T^\Pi]\le&space;t^\mathrm{max}\nonumber&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{\Pi}{\text{minimize&space;}}&space;E[P^\Pi]&space;\\&space;\text{subject&space;to&space;}&space;E[T^\Pi]\le&space;t^\mathrm{max}\nonumber&space;\\" title="\underset{\Pi}{\text{minimize }} E[P^\Pi] \\ \text{subject to } E[T^\Pi]\le t^\mathrm{max}\nonumber \\" /></a>

can be transformed to an unconstrained weighted sum optimization using the [Lagragian
multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier), where <a href="https://www.codecogs.com/eqnedit.php?latex=\Pi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Pi" title="\Pi" /></a> is 
the set of dynamic policies, E[T] denotes mean response time, and E[P] represents mean power
consumption. The resulting Lagrangian multiplier can then be iteratively
updated by applying techniques like gradient descent to minimize the weighted sum of performance
and energy.

## Inputs
Macro cell input parameters: Idle power, busy power, service rates for its own users (blue black in the above figure) and 
users offloaded from small cells (green).
Small cell input parameters: Idle, busy, setup, and sleep power consumption values. Service rate, setup time (time it takes 
to start the small cell from a sleep state), and idle timer. The number of small cells can also be arbitrarily set.

## Outputs
The outputs are: mean response time and mean power consumption under the optimized policy.

## How to run the simulator
To run the simulator in a supercomputing environment as a batch job, see the help screen of the [automated-runs.py](simulator/automated-runs.py) by executing ./automated-runs.py -h on the commandline.
