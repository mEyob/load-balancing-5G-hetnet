## Introduction

This repo provides an optimization model for energy-aware load balancing of 
user traffic in 5G hetrogeneous networks. To this end, user request scheduling 
is modeled as a Constrained Markov Decision Process (CMDP). The network consists
of a single Macro cell and a number of small cells. Small cells provide higher
data rate to users within their coverage area, but they can also be turned off 
to save energy when demand is low. Thus the task is to develop a dynamic scheduling 
policy that creates energy saving opportunities by offloading user traffic to macro 
cell whenever possible. More specifically, the scheduling policy should minimizes 
energy consumption under a given performance constraint (mean response time).

