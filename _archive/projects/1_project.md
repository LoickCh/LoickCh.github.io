---
layout: page
title: Deep Hough Voting
description: 3D object detection
img: assets/img/project/deep_hough_votting/project_deephoughvoting.png
importance: 1
category: master
---

**Reference**: <a href="https://github.com/facebookresearch/votenet">Deep Hough Voting, C.R. Qi, K. He, L.J. Guibas, 2019.</a>

**Summary:** Automatic modeling and segmentation is a major issue in 3D computer vision. There are two classical families of methods solving it: those using surface propagation and those using surface extraction by voting. The Hough method belongs to the second category. Its principle is to perform a vote in the parameter space rather than in the point space. With the arrival of neural networks, the voting principle has been extended and used to predict bounding box for scene points in the context of 3D object detection. In this project we will study all the pipeline of a new neural network whose originality is to perform 3D bounding box estimation using only 3D point clouds and no external 2D data.

**Results:** You can find below the results obtained when running the code on the github repository. Details on the training and on the implementation can be found in the following <a href="/assets/pdf/Report_Deephoughvoting.pdf"> pdf. </a>

<div class="text-center">
    <div class="mt-3 mt-md-0">{% include figure.html path="/assets/img/project/deep_hough_votting/project_deephoughvoting_results.png" class="img-fluid rounded z-depth-1" %}</div>
    <div class="caption">Point cloud, ground-truth and prediction on some scans of the 3rd testing batch of ScannetV2.</div>
    
    <div class="mt-4 mt-md-0">{% include figure.html path="/assets/img/project/deep_hough_votting/project_deephoughvoting_results_3.png" class="img-fluid rounded z-depth-1" %}</div>
    <div class="caption">Point cloud, ground-truth and prediction on some scans of the 10th testing batch of Sun RGB-D v2.</div>
</div>