---
layout: page
title: 3D object detection.
description: Reproduce the results of the "Deep Hough Voting" neural network on ScanNet and SUNRGB-D.
img: assets/img/project_deephoughvoting.jpg
importance: 1
category: work
---

Original project: <a href="https://github.com/facebookresearch/votenet">Deep Hough Voting, C.R. Qi, K. He, L.J. Guibas, 2019.</a>

Automatic modeling and segmentation is a major application in 3D computer vision. There are two classical main families of methods, those using surface propagation and those using surface extraction by voting. The Hough method belongs to the second category. Its principle is to perform a vote in the parameter space rather than in the point space. With the arrival of neural networks, the voting principle has been extended and used to predict bounding box for scene points in the context of 3D object detection. In this project we will study all the pipeline of a new neural network whose originality is to perform 3D bounding box estimation using only 3D point clouds and no external 2D data.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_deephoughvoting_results.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Point cloud, ground-truth and prediction on some scans of the 3rd testing batch of ScannetV2.
</div>
