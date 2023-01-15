---
layout: page
title: 3D semantic segmentation
description: Adapt "KP-conv" to industrial facility point clouds.
img: assets/img/IC/profile.png
importance: 1
category: work
---

**References:** <a href="https://github.com/HuguesTHOMAS/KPConv">KP-Conv, H. Thomas, et al., 2020.</a>

**Summary:** This project is part of a <a href="https://challengedata.ens.fr/participants/challenges/year/2022" > competition </a> managed by the Data team (ENS Paris), in partnership with the Collège de France. The challenge is organised by EDF R&D. The goal to perform a semantic segmentation of a 3D point cloud. The point cloud of one EDF industrial facility digital mock-ups is composed of 45 billions of 3D points. The reconstruction work consisting of the fitting of 90 000 geometric primitives on the point cloud. To perform this task, the operators have to manually segment part of the point cloud corresponding to an equipment to then fit the suitable geometric primitive. This manual segmentation is the more tedious of the global production scheme. Therefore, EDF R&D studies solutions to perform it automatically.

**Dataset**: The boiling room was digitized with LiDAR scanners on tripods and contains 67 scanner positions. Each acquisition at one scanner position (which is called “station”) produces one point cloud of about 30 millions of points. The point clouds of the set of stations are registered in a single reference frame. Randomly subsampled point clouds will be provided. The train set contains 50 stations point cloud and the test set contains the remaining 18 stations. 

**Results:** To solve this challenge, I used KP-Conv, a deep neural network build to classify and segment 3D point clouds. I ranked first in the competition with a score of 0.9400 on the private leaderboard. Details on the training and on the implementation can be found in the following <a href="/assets/pdf/Report_IC"> pdf </a>.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/IC/results_1.png" 
        title="example image" 
        class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results obtained on the test set. Some classes are over-represented, which corresponds to the distribution of classes in the training scans.
</div>

**Ressources**: I trained the network on Google Colab pro using a P100 GPU for 10 hours.