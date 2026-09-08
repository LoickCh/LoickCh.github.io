---
layout: page
title:  NAF
description: Zero-Shot Feature Upsampling
img: assets/img/paper/2025_naf/teaser.png
importance: 1
category: PhD papers
code_url: https://github.com/valeoai/NAF
paper_url: https://arxiv.org/abs/2511.18452
year: 2026
venue: CVPR (highlight)
---

<h1 align="center"> {{page.title}} </h1>
<h3 align="center">  <a href="https://loickch.github.io/">Loick Chambon</a>&nbsp;&nbsp; <a href="https://pcouairon.github.io/">Paul Couairon</a> &nbsp;&nbsp; <a href="https://scholar.google.fr/citations?user=dOkbUmEAAAAJ&hl=fr">Eloi Zablocki</a>&nbsp;&nbsp; <a href="https://boulch.eu/">Alexandre Boulch</a>&nbsp;&nbsp; &nbsp;&nbsp; <a href="https://thome.isir.upmc.fr">Nicolas Thome</a> <a href="https://cord.isir.upmc.fr/">Matthieu Cord</a></h3>

<h3 align="center"> {{page.venue}} {{page.year}} </h3>

<div class="row justify-content-center">
  <div class="column">
      <p align="center">
      {% if page.paper_url %}<a href="{{ page.paper_url }}"><i class="far fa-file-pdf"></i> Paper</a>&nbsp;&nbsp;{% endif %}
      {% if page.code_url %}<a href="{{ page.code_url }}"><i class="fab fa-github"></i> Code</a> &nbsp;&nbsp;{% endif %}
      {% if page.website_url %}<a href="{{ page.website_url }}"><i class="fas fa-globe"></i> Website</a> &nbsp;&nbsp;{% endif %}
      </p>
  </div>
</div>

<hr>

<h2 align="center">Abstract</h2>

<p align="justify">
Vision Foundation Models (VFMs) produce downsampled spatial features, which are challenging for pixel-level tasks. Existing upsampling methods either rely on fixed classical filters (bilinear, bicubic, joint bilateral) or require learnable, VFM-specific retraining (FeatUp, LiFT, JAFAR).  
We introduce <strong>NAF</strong> — Neighborhood Attention Filtering — a zero-shot, VFM-agnostic upsampler that leverages Cross-Scale Neighborhood Attention and Rotary Position Embeddings (RoPE) to learn adaptive spatial-and-content weights guided solely by the high-resolution input image.  
NAF scales up to 2K feature maps efficiently (~18 FPS) and consistently outperforms previous methods across multiple downstream tasks, including semantic segmentation, depth estimation, zero-shot open vocabulary, and video propagation. It also demonstrates strong performance for image restoration.
</p>

<hr>

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <img src="../../assets/img/paper/2025_naf/teaser.gif" class="img-fluid rounded z-depth-1"/>
        <div class="caption">
            NAF enables zero-shot feature upsampling for any Vision Foundation Model, producing high-resolution features without retraining.
        </div>
    </div>
</div>

<h2 align="center">Results</h2>

NAF allows efficient zero-shot upsampling of any VFM features to high-resolution. It achieves state-of-the-art performance across multiple downstream tasks while remaining computationally efficient.

<div style="text-align: center; margin: 2em 0;">
    <img src="../../assets/img/paper/2025_naf/results.png" class="img-fluid rounded z-depth-1" style="display: inline-block;"/>
    <div class="caption" style="text-align: center; margin-top: 0.5em;">
        Summary of NAF performance across downstream tasks: semantic segmentation, depth estimation, open-vocabulary segmentation, and video propagation.
    </div>
</div>


<h2 align="center">BibTeX</h2>
<left>
  <pre class="bibtex-box">
@misc{chambon2025nafzeroshotfeatureupsampling,
      title={NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering}, 
      author={Loick Chambon and Paul Couairon and Eloi Zablocki and Alexandre Boulch and Nicolas Thome and Matthieu Cord},
      year={2025},
      url={https://arxiv.org/abs/2511.18452}, 
}
</pre>
</left>

<br>
