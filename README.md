# World Happiness Analysis

The Happy Planet Index (HPI) is an index of human well-being and environmental impact introduced by the New Economics Foundation in July 2006. The Happy Planet Index measures what matters: sustainable wellbeing for all. It tells us how well nations are doing at achieving long, happy, sustainable lives. 156 countries are ranked based on how happy their citizens perceive themselves to be.

Column Information:

- Score is a populated-weighted average score on a scale running from 0 to 10 that is tracked over time and compared against other countries.
- GDP per capita is in terms of Purchasing Power Parity (PPP) adjusted to constant 2011 international dollars, taken from the World Development Indicators (WDI) released by the World Bank on November 14, 2018.
- The healthy life expectancy at birth are based on data from the World Health Organization (WHO) Global Health Observatory data repository.
- Social support is the national average of the binary responses (either 0 or 1) to question “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?”
- Freedom to make life choices is the national average of binary responses to the question “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”
- Generosity is the residual of regressing the national average to the question “Have you donated money to a charity in the past month?” on GDP per capita.
- Perceptions of corruption are the average of binary answers to two questions: “Is corruption widespread throughout the government or not?” and “Is corruption widespread within businesses or not?”

## Data Preparation and Cleaning

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
df2019 = pd.read_csv('./2019.csv')
```

```python
df2019
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151</th>
      <td>152</td>
      <td>Rwanda</td>
      <td>3.334</td>
      <td>0.359</td>
      <td>0.711</td>
      <td>0.614</td>
      <td>0.555</td>
      <td>0.217</td>
      <td>0.411</td>
    </tr>
    <tr>
      <th>152</th>
      <td>153</td>
      <td>Tanzania</td>
      <td>3.231</td>
      <td>0.476</td>
      <td>0.885</td>
      <td>0.499</td>
      <td>0.417</td>
      <td>0.276</td>
      <td>0.147</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>Afghanistan</td>
      <td>3.203</td>
      <td>0.350</td>
      <td>0.517</td>
      <td>0.361</td>
      <td>0.000</td>
      <td>0.158</td>
      <td>0.025</td>
    </tr>
    <tr>
      <th>154</th>
      <td>155</td>
      <td>Central African Republic</td>
      <td>3.083</td>
      <td>0.026</td>
      <td>0.000</td>
      <td>0.105</td>
      <td>0.225</td>
      <td>0.235</td>
      <td>0.035</td>
    </tr>
    <tr>
      <th>155</th>
      <td>156</td>
      <td>South Sudan</td>
      <td>2.853</td>
      <td>0.306</td>
      <td>0.575</td>
      <td>0.295</td>
      <td>0.010</td>
      <td>0.202</td>
      <td>0.091</td>
    </tr>
  </tbody>
</table>
<p>156 rows × 9 columns</p>
</div>

```python
countries = df2019.shape[0]
print('There are {} countries in the dataset'.format(countries))
```

    There are 156 countries in the dataset

```python
df2019.columns
```

    Index(['Overall rank', 'Country or region', 'Score', 'GDP per capita',
           'Social support', 'Healthy life expectancy',
           'Freedom to make life choices', 'Generosity',
           'Perceptions of corruption'],
          dtype='object')

```python
df2019.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 156 entries, 0 to 155
    Data columns (total 9 columns):
     #   Column                        Non-Null Count  Dtype
    ---  ------                        --------------  -----
     0   Overall rank                  156 non-null    int64
     1   Country or region             156 non-null    object
     2   Score                         156 non-null    float64
     3   GDP per capita                156 non-null    float64
     4   Social support                156 non-null    float64
     5   Healthy life expectancy       156 non-null    float64
     6   Freedom to make life choices  156 non-null    float64
     7   Generosity                    156 non-null    float64
     8   Perceptions of corruption     156 non-null    float64
    dtypes: float64(7), int64(1), object(1)
    memory usage: 11.1+ KB

## Exploratory Analysis and Visualization

```python
df2019.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
      <td>156.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>78.500000</td>
      <td>5.407096</td>
      <td>0.905147</td>
      <td>1.208814</td>
      <td>0.725244</td>
      <td>0.392571</td>
      <td>0.184846</td>
      <td>0.110603</td>
    </tr>
    <tr>
      <th>std</th>
      <td>45.177428</td>
      <td>1.113120</td>
      <td>0.398389</td>
      <td>0.299191</td>
      <td>0.242124</td>
      <td>0.143289</td>
      <td>0.095254</td>
      <td>0.094538</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>2.853000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>39.750000</td>
      <td>4.544500</td>
      <td>0.602750</td>
      <td>1.055750</td>
      <td>0.547750</td>
      <td>0.308000</td>
      <td>0.108750</td>
      <td>0.047000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>78.500000</td>
      <td>5.379500</td>
      <td>0.960000</td>
      <td>1.271500</td>
      <td>0.789000</td>
      <td>0.417000</td>
      <td>0.177500</td>
      <td>0.085500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>117.250000</td>
      <td>6.184500</td>
      <td>1.232500</td>
      <td>1.452500</td>
      <td>0.881750</td>
      <td>0.507250</td>
      <td>0.248250</td>
      <td>0.141250</td>
    </tr>
    <tr>
      <th>max</th>
      <td>156.000000</td>
      <td>7.769000</td>
      <td>1.684000</td>
      <td>1.624000</td>
      <td>1.141000</td>
      <td>0.631000</td>
      <td>0.566000</td>
      <td>0.453000</td>
    </tr>
  </tbody>
</table>
</div>

```python
above_mean_score =['Above World Average' if i >= df2019.mean()['Score'] else 'Below World Average' \
                   for i in df2019.Score]
df = pd.DataFrame({'Happiness_Score' : above_mean_score})
ax = sns.countplot(x = df.Happiness_Score)
for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate(int(y), (x.mean(), y), ha='center', va='bottom')
plt.xlabel('Happiness Score')
plt.ylabel('Number of Countries')
plt.title('Number of Countries based on Happiness Score Average', color = 'blue', fontsize = 15)
plt.show()
```

![png](images/1.png)

```python
# Relationship between GDP per capita and healthy life expectancy

df2019['GDP per capita']
sns.scatterplot(df2019['GDP per capita'], df2019['Healthy life expectancy'], hue=above_mean_score, s=80);
```

![png](images/2.png)

```python
# Relationship between GDP per capita and freedom to make life choices

df2019['GDP per capita']
sns.scatterplot(df2019['GDP per capita'], df2019['Freedom to make life choices'], hue=above_mean_score, s=80);
```

![png](images/3.png)

```python
# Relationship between GDP per capita and freedom to make life choices

sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

axes[0, 0].plot(df2019.Score, df2019['GDP per capita'], 'r')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('GDP per capita')
axes[0, 0].set_title('GDP per capita vs Score')

axes[0, 1].plot(df2019.Score, df2019['Social support'], 'g')
axes[0, 1].set_xlabel('Score')
axes[0, 1].set_ylabel('Social support')
axes[0, 1].set_title('Social support vs Score')

axes[0, 2].plot(df2019.Score, df2019['Healthy life expectancy'], 'b')
axes[0, 2].set_xlabel('Score')
axes[0, 2].set_ylabel('Healthy life expectancy')
axes[0, 2].set_title('Healthy life expectancy vs Score')

axes[1, 0].plot(df2019.Score, df2019['Freedom to make life choices'], 'y')
axes[1, 0].set_xlabel('Score')
axes[1, 0].set_ylabel('Freedom to make life choices')
axes[1, 0].set_title('Freedom to make life choices vs Score')

axes[1, 1].plot(df2019.Score, df2019['Generosity'], 'm')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('Generosity')
axes[1, 1].set_title('Generosity')

axes[1, 2].plot(df2019.Score, df2019['Perceptions of corruption'], 'k')
axes[1, 2].set_xlabel('Score')
axes[1, 2].set_ylabel('Perceptions of corruption')
axes[1, 2].set_title('Perceptions of corruption vs Score')

plt.tight_layout(pad=2)
plt.show()
```

![png](images/4.png)

```python
# Correlation between variables

f,ax = plt.subplots(figsize = (12, 12))
sns.heatmap(df2019.corr(), annot = True, linewidths = 0.1, fmt = '.1f', ax = ax, square = True);
```

![png](images/5.png)

## Asking and Answering Questions

```python
# Which 5 countries have the top score?

df = df2019.sort_values(by=['Score'], ascending=False)
df.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('The 5 countries with the top score are: {}'.format(', '.join(df['Country or region'].head(5))))
```

    The 5 countries with the top score are: Finland, Denmark, Norway, Iceland, Netherlands

```python
# Which 5 countries have the top GDP per capita?

df = df2019.sort_values(by=['GDP per capita'], ascending=False)
df.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>Qatar</td>
      <td>6.374</td>
      <td>1.684</td>
      <td>1.313</td>
      <td>0.871</td>
      <td>0.555</td>
      <td>0.220</td>
      <td>0.167</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Luxembourg</td>
      <td>7.090</td>
      <td>1.609</td>
      <td>1.479</td>
      <td>1.012</td>
      <td>0.526</td>
      <td>0.194</td>
      <td>0.316</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>Singapore</td>
      <td>6.262</td>
      <td>1.572</td>
      <td>1.463</td>
      <td>1.141</td>
      <td>0.556</td>
      <td>0.271</td>
      <td>0.453</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>United Arab Emirates</td>
      <td>6.825</td>
      <td>1.503</td>
      <td>1.310</td>
      <td>0.825</td>
      <td>0.598</td>
      <td>0.262</td>
      <td>0.182</td>
    </tr>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>Kuwait</td>
      <td>6.021</td>
      <td>1.500</td>
      <td>1.319</td>
      <td>0.808</td>
      <td>0.493</td>
      <td>0.142</td>
      <td>0.097</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('The 5 countries with the top GDP per capita are: {}'.format(', '.join(df['Country or region'].head(5))))
```

    The 5 countries with the top GDP per capita are: Qatar, Luxembourg, Singapore, United Arab Emirates, Kuwait

```python
# Which 5 countries have the top social support?

df = df2019.sort_values(by=['Social support'], ascending=False)
df.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>New Zealand</td>
      <td>7.307</td>
      <td>1.303</td>
      <td>1.557</td>
      <td>1.026</td>
      <td>0.585</td>
      <td>0.330</td>
      <td>0.380</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('The 5 countries with the top social support are: {}'.format(', '.join(df['Country or region'].head(5))))
```

    The 5 countries with the top social support are: Iceland, Finland, Norway, Denmark, New Zealand

```python
# Which 5 countries have the top healthy life expectancy?

df = df2019.sort_values(by=['Healthy life expectancy'], ascending=False)
df.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>Singapore</td>
      <td>6.262</td>
      <td>1.572</td>
      <td>1.463</td>
      <td>1.141</td>
      <td>0.556</td>
      <td>0.271</td>
      <td>0.453</td>
    </tr>
    <tr>
      <th>75</th>
      <td>76</td>
      <td>Hong Kong</td>
      <td>5.430</td>
      <td>1.438</td>
      <td>1.277</td>
      <td>1.122</td>
      <td>0.440</td>
      <td>0.258</td>
      <td>0.287</td>
    </tr>
    <tr>
      <th>57</th>
      <td>58</td>
      <td>Japan</td>
      <td>5.886</td>
      <td>1.327</td>
      <td>1.419</td>
      <td>1.088</td>
      <td>0.445</td>
      <td>0.069</td>
      <td>0.140</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>Spain</td>
      <td>6.354</td>
      <td>1.286</td>
      <td>1.484</td>
      <td>1.062</td>
      <td>0.362</td>
      <td>0.153</td>
      <td>0.079</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Switzerland</td>
      <td>7.480</td>
      <td>1.452</td>
      <td>1.526</td>
      <td>1.052</td>
      <td>0.572</td>
      <td>0.263</td>
      <td>0.343</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('The 5 countries with the top healthy life expectancy are: {}'.format(', '.join(df['Country or region'].head(5))))
```

    The 5 countries with the top healthy life expectancy are: Singapore, Hong Kong, Japan, Spain, Switzerland

```python
# Which 5 countries have the top freedom to make life choices?

df = df2019.sort_values(by=['Freedom to make life choices'], ascending=False)
df.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>Uzbekistan</td>
      <td>6.174</td>
      <td>0.745</td>
      <td>1.529</td>
      <td>0.756</td>
      <td>0.631</td>
      <td>0.322</td>
      <td>0.240</td>
    </tr>
    <tr>
      <th>108</th>
      <td>109</td>
      <td>Cambodia</td>
      <td>4.700</td>
      <td>0.574</td>
      <td>1.122</td>
      <td>0.637</td>
      <td>0.609</td>
      <td>0.232</td>
      <td>0.062</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>United Arab Emirates</td>
      <td>6.825</td>
      <td>1.503</td>
      <td>1.310</td>
      <td>0.825</td>
      <td>0.598</td>
      <td>0.262</td>
      <td>0.182</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('The 5 countries with the top freedom to make life choices are: {}'.format(', '.join(df['Country or region'].head(5))))
```

    The 5 countries with the top freedom to make life choices are: Uzbekistan, Cambodia, Norway, United Arab Emirates, Finland

```python
# Which 5 countries have the top generosity?

df = df2019.sort_values(by=['Generosity'], ascending=False)
df.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>130</th>
      <td>131</td>
      <td>Myanmar</td>
      <td>4.360</td>
      <td>0.710</td>
      <td>1.181</td>
      <td>0.555</td>
      <td>0.525</td>
      <td>0.566</td>
      <td>0.172</td>
    </tr>
    <tr>
      <th>91</th>
      <td>92</td>
      <td>Indonesia</td>
      <td>5.192</td>
      <td>0.931</td>
      <td>1.203</td>
      <td>0.660</td>
      <td>0.491</td>
      <td>0.498</td>
      <td>0.028</td>
    </tr>
    <tr>
      <th>146</th>
      <td>147</td>
      <td>Haiti</td>
      <td>3.597</td>
      <td>0.323</td>
      <td>0.688</td>
      <td>0.449</td>
      <td>0.026</td>
      <td>0.419</td>
      <td>0.110</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>Malta</td>
      <td>6.726</td>
      <td>1.300</td>
      <td>1.520</td>
      <td>0.999</td>
      <td>0.564</td>
      <td>0.375</td>
      <td>0.151</td>
    </tr>
    <tr>
      <th>120</th>
      <td>121</td>
      <td>Kenya</td>
      <td>4.509</td>
      <td>0.512</td>
      <td>0.983</td>
      <td>0.581</td>
      <td>0.431</td>
      <td>0.372</td>
      <td>0.053</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('The 5 countries with the top generosity are: {}'.format(', '.join(df['Country or region'].head(5))))
```

    The 5 countries with the top generosity are: Myanmar, Indonesia, Haiti, Malta, Kenya

```python
# Which 5 countries have the top perceptions of corruption?

df = df2019.sort_values(by=['Perceptions of corruption'], ascending=False)
df.head(5)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>Singapore</td>
      <td>6.262</td>
      <td>1.572</td>
      <td>1.463</td>
      <td>1.141</td>
      <td>0.556</td>
      <td>0.271</td>
      <td>0.453</td>
    </tr>
    <tr>
      <th>151</th>
      <td>152</td>
      <td>Rwanda</td>
      <td>3.334</td>
      <td>0.359</td>
      <td>0.711</td>
      <td>0.614</td>
      <td>0.555</td>
      <td>0.217</td>
      <td>0.411</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>New Zealand</td>
      <td>7.307</td>
      <td>1.303</td>
      <td>1.557</td>
      <td>1.026</td>
      <td>0.585</td>
      <td>0.330</td>
      <td>0.380</td>
    </tr>
  </tbody>
</table>
</div>

```python
print('The 5 countries with the top perceptions of corruption are: {}'.format(', '.join(df['Country or region'].head(5))))
```

    The 5 countries with the top perceptions of corruption are: Singapore, Rwanda, Denmark, Finland, New Zealand

## Inferences and Conclusion

The 2019 world happiness report features the happiness score averaged over the years 2016–2018. As per the 2019 Happiness Index, Finland is the happiest country in the world. Denmark, Norway, Iceland and Netherlands hold the next top positions.

The charts shown above implies that most developed countries have a higher happiness score compared to others, aside from a higher GDP, social support, life expectancy and freedom. The only 2 variables without a direct correlation with the score is the generosity and perceptions of corruption variables, although only a handful of countries with a high happiness score have a spike in the perception of corruption, such as Singapore.

Some might argue that the ranking results are counterintuitive when it come to certain dimensions, such as measuring unhappiness using the number of suicides. Some of the countries which are ranked among the top happiest countries in the world will also feature among the top with the highest suicide rates in the world.

There are no hard and fast indicators to determine happiness, thus it is advisable to take the report with a pinch of salt. Nevertheless, the world happiness report still has its usefulness.
