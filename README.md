# Lillian_Nguyen _Python_Portfolio
This is the portfolio of python code that I learned during BISC450C. 

# Welcome to Computational Biolgoy of Python. 
## First Jupter Notebook (1 & 2)
This is the first notebook we made.

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(style="darkgrid")
```

```python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```

```python
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>


```python
df.tail()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object

```python
non_numberic_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numberic_profits].head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(df.profit[non_numberic_profits])
```




    {'N.A.'}




```python
len(df.profit[non_numberic_profits])
```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numberic_profits], bins=range(1955, 2006))
```


![png](output_10_0.png)



```python
df = df.loc[~non_numberic_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x=0, y=0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


![png](output_15_0.png)



```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (millions)')
```


![png](output_16_0.png)



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha=0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols=2)
title = 'Increase in mean and std Fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14, 4)
fig.tight_layout()
```


![png](output_17_0.png)



```python

```

## Python Fundamentals
```python
3 + 5 * 4
```




    23




```python
weight_kg = 60
```


```python
weight_kg = 60.3
```


```python
patient_id = '001'
```


```python
weight_lb = 2.2 * weight_kg
```


```python
patient_id = 'inflam_' + patient_id
```


```python
print(weight_lb)
print(patient_id)
```

    132.66
    inflam_001



```python
print(patient_id, 'weight in kilograms:', weight_kg)
```

    inflam_001 weight in kilograms: 60.3



```python
print(type(60.3))
print(type(patient_id))
```

    <class 'float'>
    <class 'str'>



```python
print('weight in pounds:', 2.2 * weight_kg)
```

    weight in pounds: 132.66



```python
print(weight_kg)
```

    60.3



```python
weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0



```python
# There are 2.2 pounds per kilogram
weight_lb = 2.2 * weight_kg
print('weight in kilograms:', weight_kg, 'and in pounds:', weight_lb)
```

    weight in kilograms: 65.0 and in pounds: 143.0



```python
weight_kg = 100.0
print('weight in kilograms is now:', weight_kg, 'and weight in pounds is still:', weight_lb)
```

    weight in kilograms is now: 100.0 and weight in pounds is still: 143.0



```python
mass = 47.5
age = 122
mass = mass * 2.0
age = age - 20
```


```python
first, second = 'Grace', 'Hopper'
third, fourth = second, first
print(third, fourth)
```

    Hopper Grace



```python
planet = 'Earth'
apples = 5
distance = 10.5
```


```python
print(type(planet))
print(type(apples))
print(type(distance))
```

    <class 'str'>
    <class 'int'>
    <class 'float'>

## Analyzing Data (1, 2, & 3)
```python
import numpy
```


```python
numpy.loadtxt(fname='inflammation-01.csv', delimiter=',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(fname='inflammation-01.csv', delimiter=',')
```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(type(data))
```

    <class 'numpy.ndarray'>



```python
print(data.dtype)
```

    float64



```python
print(data.shape)
```

    (60, 40)



```python
print('first value in data:', data[0, 0])
```

    first value in data: 0.0



```python
print('middle value in data:', data[29, 19])
```

    middle value in data: 16.0



```python
print(data[0:4, 0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10, 0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[:3, 36:]
print('small is:')
print(small)
```

    small is:
    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
print(numpy.mean(data))
```

    6.14875



```python
import time
print(time.ctime())
```

    Wed May 10 17:18:08 2023



```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)

print('maximum inflammation:', maxval)
print('minimum inflammation:', minval)
print('standard deviation:', stdval)
```

    maximum inflammation: 20.0
    minimum inflammation: 0.0
    standard deviation: 4.613833197118566



```python
patient_0 = data[0, :] # 0 on the first axis (rows), everything on the second (columns)
print('maximum inflammation for patient 0:', numpy.amax(patient_0))
```

    maximum inflammation for patient 0: 18.0



```python
print('maximum inflammation for patient 2:', numpy.amax(data[2, :]))
```

    maximum inflammation for patient 2: 19.0



```python
print(numpy.mean(data, axis=0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
print(numpy.mean(data, axis=0).shape)
```

    (40,)



```python
print(numpy.mean(data, axis=1))
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]



```python
element = 'oxygen'
print('first three characters:', element[0:3])
print('last three characters:', element[3:6])
```

    first three characters: oxy
    last three characters: gen



```python
element[:4]
```




    'oxyg'




```python
element[4:]
```




    'en'




```python
element[:]
```




    'oxygen'




```python
element[-1]
```




    'n'




```python
element[-2]
```




    'e'




```python
element[1:-1]
```




    'xyge'




```python
element = 'oxygen'
print('last three characters:', element[-3:])
element = 'carpentry'
print('last three characters:', element[-3:])
element = 'clone'
print('last three characters:', element[-3:])
element = 'hi'
print('last three characters:', element[-3:])
```

    last three characters: gen
    last three characters: try
    last three characters: one
    last three characters: hi



```python
element[3:3]
```




    ''




```python
data[3:3, 4:4]
```




    array([], shape=(0, 0), dtype=float64)




```python
data[3:3, :]
```




    array([], shape=(0, 40), dtype=float64)




```python
import numpy

A = numpy.array([[1,2,3], [4,5,6], [7, 8, 9]])
print('A = ')
print(A)

B = numpy.hstack([A, A])
print('B = ')
print(B)

C = numpy.vstack([A, A])
print('C = ')
print(C)
```

    A = 
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    B = 
    [[1 2 3 1 2 3]
     [4 5 6 4 5 6]
     [7 8 9 7 8 9]]
    C = 
    [[1 2 3]
     [4 5 6]
     [7 8 9]
     [1 2 3]
     [4 5 6]
     [7 8 9]]



```python
D = numpy.hstack((A[:, :1], A[:, -1:]))
print('D = ')
print(D)
```

    D = 
    [[1 3]
     [4 6]
     [7 9]]



```python
D = numpy.delete(arr=A, obj=1, axis=1)
print('D = ')
print(D)
```

    D = 
    [[1 3]
     [4 6]
     [7 9]]



```python
patient3_week1 = data[3, :7]
print(patient3_week1)
```

    [0. 0. 2. 0. 4. 2. 2.]



```python
[ 0 - 0, 2 - 0, 0 - 2, 4 - 0, 2 - 4, 2 - 2 ]
```




    [0, 2, -2, 4, -2, 0]




```python
numpy.diff(patient3_week1)
```




    array([ 0.,  2., -2.,  4., -2.,  0.])




```python
numpy.diff(data, axis=1)
```




    array([[ 0.,  1.,  2., ...,  1., -3.,  0.],
           [ 1.,  1., -1., ...,  0., -1.,  1.],
           [ 1.,  0.,  2., ...,  0., -1.,  0.],
           ...,
           [ 1.,  0.,  0., ..., -1.,  0.,  0.],
           [ 0.,  0.,  1., ..., -2.,  2., -2.],
           [ 0.,  1., -1., ..., -2.,  0., -1.]])




```python
numpy.amax(numpy.diff(data, axis=1), axis=1)
```




    array([ 7., 12., 11., 10., 11., 13., 10.,  8., 10., 10.,  7.,  7., 13.,
            7., 10., 10.,  8., 10.,  9., 10., 13.,  7., 12.,  9., 12., 11.,
           10., 10.,  7., 10., 11., 10.,  8., 11., 12., 10.,  9., 10., 13.,
           10.,  7.,  7., 10., 13., 12.,  8.,  8., 10., 10.,  9.,  8., 13.,
           10.,  7., 10.,  8., 12., 10.,  7., 12.])




```python
numpy.amax(numpy.absolute(numpy.diff(data, axis=1)), axis=1)
```




    array([12., 14., 11., 13., 11., 13., 10., 12., 10., 10., 10., 12., 13.,
           10., 11., 10., 12., 13.,  9., 10., 13.,  9., 12.,  9., 12., 11.,
           10., 13.,  9., 13., 11., 11.,  8., 11., 12., 13.,  9., 10., 13.,
           11., 11., 13., 11., 13., 13., 10.,  9., 10., 10.,  9.,  9., 13.,
           10.,  9., 10., 11., 13., 10., 10., 12.])


## Storing Values in Lists

