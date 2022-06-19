---
layout: page
title: Find correlations between objects and their effects.
description: Reproduce the results of the "Omnimatte" neural network on the DAVIS dataset and on Youtube videos.
img: assets/img/project_omnimatte.jpg
importance: 1
category: work
---

Original project: <a href="https://github.com/erikalu/omnimatte">Omnimatte: Associating Objects and Their Effects in Video, E. Lu, F. Cole, et al., 2021.</a>

Video inpainting is the task of reconstructing missing pixels in a video. It is an important problem in computer vision and an essential feature in many imaging and graphics applications, e.g. object removal, image restoration, manipulation, retargeting, image composition and rendering. While image inpainting is an almost solved problem, video inpainting is more difficult to solve as approaches are often unable to maintain the sharpness of the edges and create blurry effects while being unable to remove the correlation effect of an object. In addition, some suffer from temporal coherence. Although modern approaches overcome some of these problems, most of them require a complex input mask, cannot handle multiple deletion and are unable to remove correlations associated with an object. Recently, a paper has found a new way to combine objects and their effects to create masks containing subjects and effects in a self-supervised manner using only masks and coarse segmentation images. It does this by decomposing a video into a set of RGBA layers representing the appearance of different objects and their effects in the video. Although this requires training one model per video, it can lead to many applications.


<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_omnimatte_results1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Our results.
</div>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/project_omnimatte_results1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Results from the paper.
</div>
