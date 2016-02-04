#Adaptive Resonance Theory Neural Networks

author: Aman Ahuja | github.com/amanahuja | twitter: @amanqa

## Overview

ART neural architectures are self-organizing systems. They may operate in
unsupervised or semi-supervised modes, categorizing an input pattern into
categories.

Basic ART architecture consists of an input layer (F0), a processing
interface layer (F1) and an output layer (F2). F1 and F2 units are
connected by two sets of weights: bottom-up weights b[ij] and
top-down weighs t[ji].

```

      F0        F1                   F2
   +------+  +------+            +--------+
   |      |  |      |            |        |
   |  S1  |  |  X1  |    bij     |   Y1   |
   |      |  |      | ---------> |        |
   |  S2  |  |  X2  |            |        |
   |      |  |      |            |   Y2   |
   |  S3  |  |  X3  |    tji     |        |
   |      |  |      | <--------- |        |
   |  S4  |  |  X4  |            |   Yj   |
   |      |  |      |            |        |
   |  Si  |  |  Xi  |            |        |
   +------+  +------+            +--------+
   input     interface           cluster units
   layer     layer               output layer

[created with http://asciiflow.com/]
```

When presented with an input pattern, the network identifies a candidate
cluster unit in F2, and, passing a threshold test, will update weights
for this unit. This process may occur several times for a single
presentation of an input pattern, until desired stability is reached.
This process is the "resonance" for which ART is named.

## Sources
The following material were instrumental in this project: 

  - Grossberg and Carpenter
  - https://github.com/rougier/neural-networks/blob/master/art1.py
  - Fausett, Laurene V. "Fundamentals of Neural Networks: Architectures ..."
  - Grossberg, http://www.scholarpedia.org/article/Adaptive_resonance_theory
  - https://en.m.wikipedia.org/wiki/Adaptive_resonance_theory

## Purpose

These modules are intended for demonstration and learning. They favor
elucidation and interpretability over efficiency or scalability. There
is no intention to use this code in any production environment. 

---- 
 
## Included

Included in this repository:
  - ART1: ART with binary inputs
  - ART2: ART with continuous inputs
  - Helper functions for preprocessing, etc.

To-do:
  - LA-PART1: Lateral Adaptive Priming ART;
     Two coupled fuzzy ARTS for the semi-supervised case.
  - unit tests

Won't-do
  - FART: Fuzzy logic + ART
  - LAPART2: improvement on LAPART1
  - ART3

## Requirements

  - python 2.7
  - numpy

## installation and usage
[todo]
