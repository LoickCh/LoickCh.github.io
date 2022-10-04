---
layout: post
title:  HackerRank
date:   2022-09-29
description: What did I learn solving all Hackerrank's Python questions ?
tags: python
categories: sample-posts
---

State: in progress

<p> <br> </p>

# Collections

- deque:

In the official Python documentation, collections.deque() is defined as a 
list-like container with fast appends and pops on either end. It has an approximate
O(1) complexity for such operations. It supports a *maxlen* argument. If specify, 
once a bounded length deque is full, when new items are added, a corresponding 
number of items are discarded from the opposite end. 

It has many useful methods such as: *extendleft(), appendleft(), popleft(), 
remove(), rotate()*, etc.

# Basic data types

- eval(): 

According to <a href="https://realpython.com/python-eval-function/"> Real Python</a>, 
Python’s eval() allows you to evaluate arbitrary Python expressions from a 
string-based or compiled-code-based input.

It is used in the <a href="https://www.hackerrank.com/challenges/python-lists/submissions/code/293800673">Lists </a> 
challenge to evaluate different list-methods such as *.insert*, *.remove*, *.pop*, *.reverse*, etc. 
A possible answer is the following code:

```python 
if __name__ == '__main__':
    N = int(input())
    List=[]
    for _ in range(N):
        command, *args=input().split(' ')
        if command != 'print':
            eval(f'List.{command}({",".join(args)})')
        else:
            print(List)
```

*Tips:* It is intersting to notice how we combined f-strings with personalized command and 
string-to-list conversion to add arguments.

*Warning:* eval() must always be used passing globals and locals dictionnaries 
specifying which expressions can or can not be used to prevent a user from bad 
manipulations such as *os.system('rm -rf *')*. For more details, click on 
the following <a href=https://www.programiz.com/python-programming/methods/built-in/eval>website</a>.


<p> <br> </p>

# String

- str.swapcase():

Simple method returning a copy of the string with uppercase characters converted 
to lowercase and vice versa. It behaves as:

```python 
def swap_case(s):
    new_s=''
    for _ in s:
        if _==_.upper():
            new_s+=_.lower()
        else:
            new_s+=_.upper()
    return new_s
```

- str.isalnum(), str.isalpha(), str.isdigit(), str.islower(), str.isupper():

Check whether each character of the string is alphanumeric (a-z, A-Z, 0-9), 
alphabetical (a-z, A-Z), is a digit (0-9), is lowercase, is uppercase.

- str.capitalize():
  
Transforms the first character of the string into an upper case character. 
It is equivalent to:

```python 
def capitalize(s):
    splitted=s.split(' ')
    return ' '.join( list(map(lambda s: s[:1].upper() + s[1:], splitted)) )
```

<p> <br> </p>

# Regular expressions

According to the Python library, the *re* module provides regular expression 
matching operations similar to those found in Perl. Regular expressions can help 
in many situations when we have to verify if something is contained in a string or 
more generally, when a string follows a certain pattern.

- re.findall():

The expression re.findall() returns all the non-overlapping matches of patterns 
in a string as a list of strings.

In a challenge, we had to find every pattern containing two or more vowels lying between
two consonnants. To look behind a pattern, we use "?<=", to look ahead we use "?=".
Moreover, since we have to find two or more vowels, we need to specify "{2,}". 
A possible solution might be the following one:

```python
import re

s=input()
res=re.findall(r'(?<=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])([aeiouAEIUO]{2,})(?=[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])', s)
```

- re.finditer():
The expression re.finditer() returns an iterator yielding MatchObject instances 
over all non-overlapping matches for the re pattern in the string. 

    - re.start() and re.end(): MatchObject comes with additional usefull features such as *.start()* and *.end()*
These expressions return the indices of the start and end of the substring 
matched by the group. They are usefull when we have to find the indices of the 
start and end of a substring in a string.