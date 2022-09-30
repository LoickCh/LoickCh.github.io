---
layout: post
title:  Common errors
date:   2022-09-29
description: A common list of common mistakes and bad understandings.
tags: python
categories: sample-posts
---


<p> <br> </p>

# Table of content

* Tuple.
* Dict.

# Tuple

## Definition
According to the Python documentation, a tuple is a number of values separated by commas and enclosed in parentheses. 

E.g: Imagine we are a teacher that would like to represent students using a list of tuples containing their name, their age and their grades. Each individual tuple looks like the following:

```python
>>> name,age,grades='Name',22,[15,18,14]
>>> tuple_=(name, age, grades)
```

<p> <br> </p>

## Immutable lists ?

We often think tuples as immutable lists. In fact, the immutability applies to the references contained in it. It implies that, if a tuple contains a mutable reference that is modified, the value is also changed inside the tuple.

E.g: After an exam, a new grade is added to *grades*. Then the tuple is modified.


```python
>>> id(tuple_)
139927734598592
>>> grades.append(20)
>>> id(tuple_)
139927734598592
>>> tuple_
('Name', 22, [15, 18, 14, 20])
```

Hint: the best way to know if a tuple is mutable or not is to try to hash it. If it is hashable, it contains only immutable objects.



<p> <br> </p>

## In-place operations

We know that tuples are immutable in references. A consequence is that we can not modify tuples in-place with "+=", "*=", etc. 

E.g: When we add a field representing the mother tongue, Python raises an error.

```python
>>> tuple_
('Name', 22, [15, 18, 14, 20])
>>> tuple_+='French'
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: can only concatenate tuple (not "str") to tuple
```

However, there exist tricky behaviours when we try to modify mutable objects. Indeed, Python applies in-place operations to the pointed object and then try to modify the tuple reference. The *TypeError* appends in a second time so the pointed object is already modified!

E.g: When we add a grade by modifying in place the grades (the list referenced in the tuple), the list is modified but we have an assignment issue due to tuple immutability.

```python
>>> tuple_[-1]+=[2]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> tuple_
('Name', 22, [15, 18, 14, 20, 2])
>>> grades
[15, 18, 14, 20, 2]
```