---
layout: page
title: 3D reconstruction
description: Reproduce the results of various papers to reconstruct 3D meshes.
img: assets/img/RecVis/project_recvis.png
importance: 1
category: work
---

**References:** <a href="https://github.com/facebookresearch/DeepSDF">Deep SDF, JJ. Park, et al., 2019.</a> <a href="https://github.com/autonomousvision/occupancy_networks"> Occupancy Networks, L Mescheder, 2018.</a> <a href="https://github.com/autonomousvision/shape_as_points"> Shape as Points, S. Peng et al, 2021.</a>

**Summary:**
Several modern methods to reconstruct a 3D meshcan be grouped into two categories: those using implicit representations such as an occupancy functions or a signed distance functions and those using an hybrid representation (e.g solving the Poisson equation). We find out that these approaches allow to learn complex shapes and yield similar results on a specific class of the ShapeNet dataset.

**Results:** You can find below the results obtained after training DeepSDF, Occupancy network and Shape as Points on the sofa category of ShapeNet. Details on the training and on the implementation can be found in the following <a href="/assets/pdf/Report_DeepSDF.pdf"> pdf. </a>

<div class="row justify-content-md-center">
    <div class="col-sm">
    </div>
    <div class="col-auto-4">
        {% include figure.html path="assets/img/RecVis/project_recvis_results.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm">
    </div>
</div>
<div class="caption">
    Groundtruth, DeepSDF, ONET and SAP low noise.
</div>

**Notes**: 
* The project is quite GPU intensive, the database is also heavy, so networks cannot be fully trained on Colab. The free trial of Google Cloud could be a solution.
* The DeepSDF preprocessing is not easy to reproduce and configure and causes several installation problems. We preferred to use a third party package.