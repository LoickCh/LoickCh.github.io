---
layout: page
title: OccNet
description:  3D reconstruction
img: assets/img/project/occupancy_net/project_recvis.png
importance: 3
category: master
---

**References:** <a href="https://github.com/facebookresearch/DeepSDF">Deep SDF, JJ. Park, et al., 2019.</a> <a href="https://github.com/autonomousvision/occupancy_networks"> Occupancy Networks, L Mescheder, 2018.</a> <a href="https://github.com/autonomousvision/shape_as_points"> Shape as Points, S. Peng et al, 2021.</a>

**Summary:**
Several modern methods to reconstruct a 3D meshcan be grouped into two categories: those using implicit representations such as an occupancy functions or a signed distance functions and those using an hybrid representation (e.g solving the Poisson equation). We find out that these approaches allow to learn complex shapes and yield similar results on a specific class of the ShapeNet dataset.

**Results:** You can find below the results obtained after training DeepSDF, Occupancy network and Shape as Points on the sofa category of ShapeNet. Details on the training and on the implementation can be found in the following <a href="/assets/pdf/Report_DeepSDF.pdf"> pdf. </a>

<div class="text-center">
    <div class="mt-3 mt-md-0">{% include figure.html path="assets/img/project/occupancy_net/project_recvis_results.png" class="img-fluid rounded z-depth-1" %}</div>
    <div class="caption">Groundtruth, DeepSDF, ONET and SAP low noise.</div>
</div>

**Notes**: 
* The project is quite GPU intensive, the database is also heavy, so networks cannot be fully trained on Colab. The free trial of Google Cloud could be a solution.
* The DeepSDF preprocessing is not easy to reproduce and configure and causes several installation problems. We preferred to use a third party package.