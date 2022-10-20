---
layout: post
title:  HackerRank (Python)
date:   2022-10-03
description: What did I learn solving all Hackerrank's Python questions ?
tags: python
categories: challenge
---

### Progress
- Actual rank: 11.377
- Problem solved: 95% 

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

<p> <br> </p>

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

# Itertools

- groupby():

Quoting from the official Python <a href=https://docs.python.org/3/library/itertools.html#itertools.groupby> documentation </a>,
*groupby()* is used to:

"Make an iterator that returns consecutive keys and groups from the iterable. The key is a function computing a key value for each element. If not specified or is None, key defaults to an identity function and returns the element unchanged. Generally, the iterable needs to already be sorted on the same key function.

The operation of groupby() is similar to the uniq filter in Unix. It generates a break or new group every time the value of the key function changes (which is why it is usually necessary to have sorted the data using the same key function). That behavior differs from SQL’s GROUP BY which aggregates common elements regardless of their input order."

In the "Compress the string challenge", we have to group the consecutive occurences of 
string characters. One possible solution is to do this:

```python
from itertools import groupby

S=input()
print(*[ (len(list(g)),int(k)) for k,g in groupby(S)])
```

- combinations():

In "Iterable and iterators" challenge, we have to find the probability that at 
least one of the K indices selected contains the letter:'a'. To do this, we need
to use itertools.combinations() that unlike itertools.permutations(), does not take
into account the order of tuple elements. One possible solution for this challenge
is the following:

```python 
from itertools import combinations

N=int(input())
elements=input().split()
K=int(input())

indices=[i for i in range(N) if elements[i]=='a']
comb=list(combinations(range(N),K))
print( len([ i for i in comb if len(set(i) & set(indices)) > 0]) / len(comb))
```

<p> <br> </p>

# Maths

- complex() and cmath:

cmath is a module providing mathematical functions for complex numbers. It can be used
to get module of phase of complex numbers using *abs()* and *cmath.phase()* in order 
to convert numbers from and to polard coordinates.

```python
from cmath import phase

z=complex(input())
print(abs(z), phase(z)) 
```

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

- re.search():

Scan through string looking for the first location where the regular expression 
pattern produces a match, and return a corresponding match object. 

It can be used in the "Detect floating point number" challenge where we need to 
find if a the next string represents a floating point number.

```python 
import re

N=int(input())
for inp in range(N):
    # ^: beginning of the string
    # $: end of the string
    # []: character class
    # \d: matches digit from 0 to 9
    # *: any quantity
    print(True if re.search(r'^[-+]?\d*[.]\d*$', input()) else False )
```

It can be used in the "Validate list of email address with filter" challenge where
we need to determine if an email is valid or not:

```python
import re

def fun(s):
    # \w: Matches Unicode word characters.
    # +: 1 or more repetitions of the preceding RE.
    # {1,3}: at least 1, up to 3.
    # $: end of the string.
    return True if re.search("^[\w-]+@[a-zA-Z0-9]+\.[a-zA-Z]{1,3}$", s) else False
```

It can be used on the "Validating credit card" challenge where we need to guess
if the credit card is valid or not.

```python
import re

n=int(input())
for _ in range(n):
    cc=input()
    # ^[456]: begins with 4,5 or 6.
    # \d{3}: contains three digits
    # (-?\d{4}){3}: might contains - and have four digits. The pattern is repeated 3 times.
    # \1: refers to the first capturing group
    # (\d)(-?\1){3}: defines a digit group, then a group containing either - or a digit.
    # The pattern is repeated 3 times.
    if (re.fullmatch(r"^[456]\d{3}(-?\d{4}){3}$", cc) and \
         not re.search(r"(\d)(-?\1){3}", cc)):
        print("Valid")
    else:
        print("Invalid")
```

- re.match():

According to <a href="https://www.geeksforgeeks.org/python-re-search-vs-re-match/"> geeksforgeeks </a>,
re.match() searches only from the beginning of the string and return match object 
if found. But if a match of substring is found somewhere in the middle of the string, 
it returns none. While re.search() searches for the whole string even if the string
 contains multi-lines and tries to find a match of the substring in all the lines of string.
 
It can be used on the "Validate roman problem" challenge where we need to guess
if the roman number is valid or not.

```python
import re

# (M,{,3}): thousands from 1000 to 3000.
# (C[DM]|D?C{0,3}): either 900 CM, 400 CD or 100-300 C-CCC and 500-800 D-DCCC, etc.
regex_pattern = r"(M{,3})(C[DM]|D?C{0,3})(X[LC]|L?X{0,3})(I[VX]|V?I{0,3})$"
print(str(bool(re.match(regex_pattern, input()))))
```

- re.split():

Split the string by the matches of a regular expression. Inspired from the challenge
're.split()', we can have a usefull folder separator:

```python
>>> S='user/works/python/awesomefile.py'
>>> re.split(r'[/]',S)
['user', 'works', 'python', 'awesomefile.py']
```

- re.sub():

Return the string obtained by replacing the leftmost non-overlapping occurrences 
of pattern in string by the replacement repl. The challenge 'Regex Substitution'
asks to replace "&&" and "||" by "and" and "or". To avoid nested re.sub(), one
possible solution is to use match.group() and a replacement function.

```python
import re

# (?<= ): should begin by a blank space.
# (&&|\|\|): should contain either double & or double |
# (?<= ): should end by a blank space.
pattern = re.compile(r"(?<= )(&&|\|\|)(?= )")
ls = lambda x: 'and' if x.group(0) == "&&" else 'or'

for i in range(int(input())):
    s = input()
    m = pattern.sub(ls, s)
    print(m)

```


<p> <br> </p>

# Datetime

- datetime.strptime():

It creates a datetime object from a string representing a date and time and a 
corresponding format string. The format must follows conventions: %a represents 
abbreviated weekdays, %d represents day of the month as a zero-padded decimal number,
etc.

Using this function, the *Time delta challenge* can be solved with the following code:

```python
def time_delta(t1, t2):
    # represented pattern: Day dd Mon yyyy hh:mm:ss +xxxx
    fmt = "%a %d %b %Y %H:%M:%S %z"
    t1_dt = dt.strptime(t1, fmt)
    t2_dt = dt.strptime(t2, fmt)
    return str(abs(int((t1_dt - t2_dt).total_seconds())))
```
