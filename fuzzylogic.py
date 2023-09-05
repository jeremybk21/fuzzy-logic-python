### fuzzylogic.py
### Fuzzy Logic in Python
### Author: Jeremy B. Kimball
### Date: 2022-03-04

### A simple implementation of fuzzy logic in Python. This module contains the following classes:
### - TriangularFuzzySet
### - TrapezoidalFuzzySet
### - ZShapeFuzzySet
### - FuzzyRule
### - FuzzySystem
### - FuzzyInferenceSystem
### - AndFuzzySet
### - OrFuzzySet

### The following functions are also included:
### - plot_fuzzy_sets

import numpy as np
import matplotlib.pyplot as plt

def plot_fuzzy_sets(fuzzy_sets, x_min, x_max, n_points=100, title=None, POI=None):
    ''' Plot the membership functions of the fuzzy sets
    Parameters
    ----------
    fuzzy_sets : list
        List of fuzzy sets
    x_min : float
        Minimum value of the input variable
    x_max : float
        Maximum value of the input variable
    n_points : int, optional    
        Number of points to plot the membership functions, by default 100
    title : str, optional
        Title of the plot, by default None
    POI : float, optional
        Point of interest, by default None
    '''
    x = np.linspace(x_min, x_max, n_points, endpoint=True)
    if POI:
        x = np.sort(np.append(x, POI))
        
    for fuzzy_set in fuzzy_sets:
        y = [fuzzy_set.membership(xi) for xi in x]
        plt.plot(x, y, label=fuzzy_set.name)
    plt.legend()
    plt.xlabel('Input')
    plt.ylabel('Membership')
    if title:
        plt.title(title)
    plt.show()

class ConstantFuzzySet:
    ''' Constant fuzzy set
    Parameters
    ----------
    value : float
        Value of the constant fuzzy set
    name : str, optional
        Name of the fuzzy set, by default None

    Methods
    -------
    membership(x)
        Calculate the membership of the fuzzy set for a given value of the input variable
    '''
    def __init__(self, value, name=None):
        if not name:
            self.name = f'ConstantFuzzySet({value})'
        else:
            self.name = name
        self.value = value
        
    def membership(self, x):
        if x == self.value:
            return 1
        else:
            return 0
        
    def x_value(self, y=None):
        return self.value

class TriangularFuzzySet:
    ''' Triangular fuzzy set
    Parameters
    ----------
    a : float
        Lower bound of the fuzzy set
    b : float
        Peak of the fuzzy set
    c : float
        Upper bound of the fuzzy set
    
    Methods
    -------
    membership(x)
        Calculate the membership of the fuzzy set for a given value of the input variable
    '''
    def __init__(self, a, b, c, name=None):
        if not name:
            self.name = f'TriangularFuzzySet({a}, {b}, {c})'
        else:
            self.name = name
        self.a = a
        self.b = b
        self.c = c
        
    def membership(self, x):
        if x <= self.a or x >= self.c:
            return 0
        elif x == self.b:
            return 1
        elif x > self.a and x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.c - x) / (self.c - self.b)

class TrapezoidalFuzzySet:
    ''' Trapezoidal fuzzy set
    Parameters
    ----------
    a : float
        Lower bound of the fuzzy set
    b : float
        Lower peak of the fuzzy set
    c : float
        Upper peak of the fuzzy set
    d : float
        Upper bound of the fuzzy set
    name : str, optional
        Name of the fuzzy set, by default None

    Methods
    -------
    membership(x)
        Calculate the membership of the fuzzy set for a given value of the input variable

    __and__(other)
        Calculate the intersection of two fuzzy sets

    __or__(other)
        Calculate the union of two fuzzy sets

    midpoint()
        Calculate the midpoint of the fuzzy set
    '''
    def __init__(self, a, b, c, d, name=None):
        if not name:
            self.name = f'TrapezoidalFuzzySet({a}, {b}, {c}, {d})'
        else:
            self.name = name
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def membership(self, x):
        if x <= self.a or x >= self.d:
            return 0
        elif x >= self.b and x <= self.c:
            return 1
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return (self.d - x) / (self.d - self.c)
        
    def __and__(self, other):
        a = max(self.a, other.a)
        b = max(self.b, other.b)
        c = min(self.c, other.c)
        d = min(self.d, other.d)
        return TrapezoidalFuzzySet(a, b, c, d)

    def __or__(self, other):
        a = min(self.a, other.a)
        b = min(self.b, other.b)
        c = max(self.c, other.c)
        d = max(self.d, other.d)
        return TrapezoidalFuzzySet(a, b, c, d)
    
    def area(self):
        return (self.b - self.a) + (self.d - self.c) + 2 * min(self.b - self.a, self.d - self.c)
    
    def midpoint(self):
        return (self.b + self.c) / 2 
    
class ZShapeFuzzySet:
    ''' Z-shaped fuzzy set
    Parameters
    ----------
    a : float
        Lower bound of the fuzzy set
    b : float
        Upper bound of the fuzzy set
    inverted : bool, optional
        Whether the fuzzy set is inverted, by default False. Inverted is essentially S-shaped.
    name : str, optional
        Name of the fuzzy set, by default None
    input_num : int, optional
        Input variable number that the set corresponds to, by default 0

    Methods
    -------
    membership(x)
        Calculate the membership of the fuzzy set for a given value of the input variable

    __and__(other)  
        Calculate the intersection of two fuzzy sets

    __or__(other)
        Calculate the union of two fuzzy sets

    midpoint()
        Calculate the midpoint of the fuzzy set

    area()  
        Calculate the area of the fuzzy set
    '''

    def __init__(self, a, b, input_num=0, inverted=False, name=None):
        if not name:
            self.name = f'ZShapeFuzzySet({a}, {b}, inverted={inverted})'
        else:
            self.name = name
        self.a = a
        self.b = b
        self.inverted = inverted
        self.input_num = input_num
        
    def membership(self, x):
        if isinstance(x, list):
            x = x[self.input_num]
        if self.inverted:
            if x <= self.a:
                return 0
            elif x >= self.b:
                return 1
            else:
                return (x - self.a) / (self.b - self.a)
        else:
            if x <= self.a:
                return 1
            elif x >= self.b:
                return 0
            else:
                return 1 - (x - self.a) / (self.b - self.a)
    
    def __and__(self, other):
        return AndFuzzySet(self, other)
    
    def __or__(self, other):
        return OrFuzzySet(self, other)

class AndFuzzySet:
    ''' Fuzzy set that represents the intersection of two fuzzy sets
    Parameters
    ----------
    set1 : FuzzySet
        First fuzzy set
    set2 : FuzzySet
        Second fuzzy set

    Methods
    -------
    membership(x)
        Calculate the membership of the fuzzy set for a given value of the input variable
    '''
    def __init__(self, set1, set2):
        self.set1 = set1
        self.set2 = set2
        self.name = f'AndFuzzySet({set1.name}, {set2.name})'
        
    def membership(self, x):
        return min(self.set1.membership(x), self.set2.membership(x))
    
class OrFuzzySet:
    ''' Fuzzy set that represents the union of two fuzzy sets
    Parameters
    ----------
    set1 : FuzzySet
        First fuzzy set
    set2 : FuzzySet
        Second fuzzy set

    Methods
    -------
    membership(x)
        Calculate the membership of the fuzzy set for a given value of the input variable
    '''
    def __init__(self, set1, set2):
        self.set1 = set1
        self.set2 = set2
        self.name = f'OrFuzzySet({set1.name}, {set2.name})'
        
    def membership(self, x):
        return max(self.set1.membership(x), self.set2.membership(x))

class FuzzyRule:
    ''' Fuzzy rule
    Parameters
    ----------
    antecedent : FuzzySet
        Antecedent of the rule
    consequent : FuzzySet
        Consequent of the rule

    Methods
    -------
    evaluate(x)
        Evaluate the strength of the rule based on the input x

    get_output()
        Get the crisp output for the rule consequent based on antecedent membership strength.

    '''
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

    def evaluate(self, x):
        return self.antecedent.membership(x)

    def get_output(self, x=None):
        if x is None:
            return self.consequent.x_value()
        else:
            return self.consequent.x_value(self.antecedent.membership(x))

class FuzzySystem:
    ''' Fuzzy system
    Parameters
    ----------
    rules : list of FuzzyRule
        List of rules of the system

    Methods
    -------
    output(x)
        Compute the crisp output of the system based on the input x. Uses Tsukamoto's method.
    '''

    def __init__(self, rules):
        self.rules = rules

    def output(self, x):
        # Evaluate strength of each rule based on input x
        strengths = [rule.evaluate(x) for rule in self.rules]
        
        # Determine output strength of the system
        output_strength = np.fmax.reduce(strengths)
        
        # Compute weighted average of midpoints of consequents of each rule
        if output_strength == 0:
            return 0
        else:
            numerator = np.sum([rule.get_output() for rule in self.rules] * np.array(strengths), axis=0)
            denominator = np.sum(strengths)
            return np.sum(numerator / denominator)