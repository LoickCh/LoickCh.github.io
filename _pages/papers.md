---
layout: page
title: papers
permalink: /papers/
description: |
  I defended my PhD in Computer Vision on **23 February 2026** at **Sorbonne University** and **Valeo.ai**, supervised by [Matthieu Cord](https://scholar.google.fr/citations?user=SpAotDcAAAAJ&hl=en), [Alexandre Boulch](https://scholar.google.com/citations?user=iJ3qFGAAAAAJ&hl=fr), and [Eloi Zablocki](https://scholar.google.com/citations?user=dOkbUmEAAAAJ&hl=fr).

  My research focused on **Vision Foundation Models** and **perception for autonomous driving**, with a particular interest in **feature upsampling**, **high-resolution visual representations**, **scalable 3D scene understanding**, and **efficient perception models** for real-world applications.
nav: true
nav_order: 1
display_categories: ["PhD papers"]
horizontal: false
---

<!-- pages/papers.md -->
<div class="papers">
{%- if site.enable_paper_categories and page.display_categories %}
  <!-- Display categorized papers -->
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  {%- assign categorized_papers = site.papers | where: "category", category -%}
  {%- assign sorted_papers = categorized_papers | sort: "importance" %}
  <!-- Generate cards for each paper -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for paper in sorted_papers -%}
      {% include papers_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for paper in sorted_papers -%}
      {% include papers.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
  {% endfor %}

{%- else -%}
<!-- Display papers without categories -->
  {%- assign sorted_papers = site.papers | sort: "importance" -%}
  <!-- Generate cards for each paper -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for paper in sorted_papers -%}
      {% include papers_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for paper in sorted_papers -%}
      {% include papers.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
{%- endif -%}
</div>
