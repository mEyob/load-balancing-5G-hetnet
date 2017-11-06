## Introduction

This repo provides a mini python optimization package for energy-aware load balancing of 
user traffic in 5G hetrogeneous networks. To this end, user request scheduling 
is modeled as a Constrained Markov Decision Process (CMDP). As depicted in the
following figure, the network consists
of a single Macro cell and a number of small cells. Small cells provide higher
data rate to users within their coverage area, but they can also be turned off 
to save energy when demand is low. Thus the task is to develop a dynamic scheduling 
policy that creates energy saving opportunities by offloading user traffic to macro 
cell whenever possible. More specifically, the scheduling policy should minimizes 
energy consumption under a given performance constraint (mean response time).

<img src="hetnet-model.png" alt="perHr" style="width: 100px; height: 100px" />

# Converting constrained optimization to unconstrained
The constrained optimization 

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{eqnarray}&space;\begin{aligned}&space;\min&space;&&space;E[P]&space;\\&space;&&space;E[T]\le&space;t^\mathrm{max}\nonumber&space;\\&space;\end{aligned}&space;\end{eqnarray}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{eqnarray}&space;\begin{aligned}&space;\min&space;&&space;E[P]&space;\\&space;&&space;E[T]\le&space;t^\mathrm{max}\nonumber&space;\\&space;\end{aligned}&space;\end{eqnarray}" title="\begin{eqnarray} \begin{aligned} \min & E[P] \\ & E[T]\le t^\mathrm{max}\nonumber \\ \end{aligned} \end{eqnarray}" /></a>
