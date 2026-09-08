---
layout: page
title:  PointBeV
description: 2D BeV Segmentation
img: assets/img/paper/2024_pointbev/teaser.png
importance: 4
category: PhD papers
year: 2024
venue: CVPR
paper_url: https://arxiv.org/abs/2312.00703
code_url: https://github.com/valeoai/PointBeV
---

<h1 align="center"> {{page.title}} </h1>
<h3 align="center"> Loïck Chambon &nbsp;&nbsp; <a href="https://eloiz.github.io">Éloi Zablocki</a> &nbsp;&nbsp; <a href="https://scholar.google.com/citations?user=QnRpMJAAAAAJ">Mickaël Chen</a> &nbsp;&nbsp; <a href="https://f-barto.github.io/">Florent Bartoccioni</a> &nbsp;&nbsp; <a href="https://ptrckprz.github.io/">Patrick Pérez</a>  &nbsp;&nbsp; <a href="https://cord.isir.upmc.fr/">Matthieu Cord</a></h3>


<h3 align="center"> {{page.venue}} {{page.year}} </h3>

<div align="center">
  <p>
    {% if page.paper_url %}
    <a href="{{ page.paper_url }}"><i class="far fa-file-pdf"></i> Paper</a>&nbsp;&nbsp;
    {% endif %}
    {% if page.code_url %}
    <a href="{{ page.code_url }}"><i class="fab fa-github"></i> Code</a> &nbsp;&nbsp;
    {% endif %}
    {% if page.blog_url %}
    <a href="{{ page.blog_url }}"><i class="fab fa-blogger"></i> Blog</a> &nbsp;&nbsp;
    {% endif %}
    {% if page.slides_url %}
    <a href="{{ page.slides_url }}"><i class="far fa-file-pdf"></i> Slides</a>&nbsp;&nbsp;
    {% endif %}
    {% if page.bib_url %}
    <a href="{{ page.bib_url}}"><i class="far fa-file-alt"></i> BibTeX</a>&nbsp;&nbsp;
    {% endif %}
  </p>
</div>

<hr>

<h2  align="center"> Abstract</h2>

<p align="justify">Bird's-eye View (BeV) representations have emerged as the de-facto shared space in driving applications, offering a unified space for sensor data fusion and supporting various downstream tasks. However, conventional models use grids with fixed resolution and range and face computational inefficiencies due to the uniform allocation of resources across all cells. To address this, we propose PointBeV, a novel sparse BeV segmentation model operating on sparse BeV cells instead of dense grids. This approach offers precise control over memory usage, enabling the use of long temporal contexts and accommodating memory-constrained platforms. PointBeV employs an efficient two-pass strategy for training, enabling focused computation on regions of interest. At inference time, it can be used with various memory/performance trade-offs and flexibly adjusts to new specific use cases. PointBeV achieves state-of-the-art results on the nuScenes dataset for vehicle, pedestrian, and lane segmentation, showcasing superior performance in static and temporal settings despite being trained solely with sparse signals. We will release our code along with two new efficient modules used in the architecture: Sparse Feature Pulling, designed for the effective extraction of features from images to BeV, and Submanifold Attention, which enables efficient temporal modeling.</p>

<hr>

<div class="text-center">
    <div class="mt-3 mt-md-0">{% include figure.html path="assets/img/paper/2024_pointbev/pointbev.png" class="img-fluid rounded z-depth-1" %}</div>
    <div class="caption">Architecture.</div>
</div>

<hr>

<h2  align="center">BibTeX</h2>
<left>
  <pre class="bibtex-box">
@inproceedings{chambon2024pointbev,
      title={PointBeV: A Sparse Approach to BeV Predictions},
      author={Loick Chambon and Eloi Zablocki and Mickael Chen and Florent Bartoccioni and Patrick Perez and Matthieu Cord},
      year={2024},
      booktitle={CVPR}
}
</pre>
</left>

<br>