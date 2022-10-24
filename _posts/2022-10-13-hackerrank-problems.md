---
layout: post
title:  HackerRank (Problems)
date:   2022-10-13
description: What did I learn solving everyday Hackerrank's algorithms challenges ?
tags: algorithms
categories: challenge
---

### Progress
- Actual rank: 262.087
- Points: 683/850.

### Training: try again

- picking numbers: try to sort the list first.
- non divisible subset: use modulos.

<p> <br> </p>

# Datetime

Datetime contains useful functions for manipulating dates in Python. In the "*Time conversion*"
challenge, we had to convert a date from a 12h format into a 24h format. It can be easily
solved using two functions: datatime.strptime and datetime.strftime.

- datetime.strptime(): read a string according to a format date and output a date 
object.
- datetime.strftime(): convert a date object to a string w.r.t format.

```python
from datetime import datetime

def timeConversion(s):
    # Write your code here
    return datetime.strptime(s,'%I:%M:%S%p').strftime('%H:%M:%S')
```

<p> <br> </p>

# Math

Math contains useful built-ins functions to perform mathematical calculus without
using numpy. In the *between two sets* challenge, we had to determine numbers such that:
- The elements of the first array are all factors of the integer being considered
- The integer being considered is a factor of all elements of the second array
To do so, we can use math.gcd and math.lcm to calculate the greatest common divisor 
between two elements and the least common multiple of multiple elements:

```python
from functools import reduce

def getTotalX(a, b):
    # Write your code here
    lcm=math.lcm(*a)
    gcd=reduce(lambda x,y: math.gcd(x,y), b)
    nbr=0
    for _ in range(lcm,gcd+1,lcm):
        if _%lcm==0 and gcd/_ == gcd//_:
            nbr += 1
    return nbr
```

<p> <br> </p>

# Bisect

The bisect module allows to insert item in sorted list while maintaining order.
The module is called bisect because it uses a basic bisection algorithm. It contains
insort (*insort_left*, *insort_right*) functions to add elements in a sorted list and locate functions to know where elements should be inserted (*bisect_right*, *bisect_left*). In the 
"Climbing the leaderboard" challenge, we have to find player ranks according to 
a list of rank. Here is a possible solution:

```python 
import bisect

def climbingLeaderboard(ranked, player):
    ranked=sorted(list(set(ranked)))
    player=sorted(player)
    records=[]
    records=list(map( lambda x: len(ranked)-bisect.bisect(ranked, x)+1, player))
    return records
```

<p> <br> </p>

# Deque

Deque  provides an O(1) time complexity for append and pop operations as compared 
to a list that provides O(n) time complexity. It contains useful functions to append
(*append*, *append_left*, *extend*, *extend_left*) and to pop (*pop*, *pop_left*, *remove*)
elements. In *Circular array rotation challenge*, we use deque for its rotate function:

```python 
from collections import deque

def circularArrayRotation(a, k, queries):
    a=deque(a)
    deque.rotate(a, k)
    res=[]
    for q in queries:
        res.append(a[q])
    return res
```