```python
!unzip "/content/bike-sharing-demand.zip"
```

    Archive:  /content/bike-sharing-demand.zip
      inflating: sampleSubmission.csv    
      inflating: test.csv                
      inflating: train.csv               
    


```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```


```python
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df = pd.concat([df_train, df_test])
```

* Data Fields
* datetime - hourly date + timestamp  
* season:
 - 1 = spring
 - 2 = summer
 - 3 = fall
 - 4 = winter
* holiday - whether the day is considered a holiday
* workingday - whether the day is neither a weekend nor holiday
* weather
 - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
 - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
 - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
 - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
* temp - temperature in Celsius
* atemp - "feels like" temperature in Celsius
* humidity - relative humidity
* windspeed - wind speed
* casual - number of non-registered user rentals initiated
* registered - number of registered user rentals initiated
* count - number of total rentals


```python
df.head()
```





  <div id="df-e41654ec-958a-4074-8225-e877796de6d1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>13.0</td>
      <td>16.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>32.0</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>27.0</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>10.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e41654ec-958a-4074-8225-e877796de6d1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e41654ec-958a-4074-8225-e877796de6d1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e41654ec-958a-4074-8225-e877796de6d1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9c46a37d-f6fd-4597-8e79-4ef992b336f0">
  <button class="colab-df-quickchart" onclick="quickchart('df-9c46a37d-f6fd-4597-8e79-4ef992b336f0')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9c46a37d-f6fd-4597-8e79-4ef992b336f0 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df.describe()
```





  <div id="df-e3bba27f-1a7c-4311-ab23-dc4b368cdcae" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
      <td>10886.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.501640</td>
      <td>0.028770</td>
      <td>0.682721</td>
      <td>1.425283</td>
      <td>20.376474</td>
      <td>23.788755</td>
      <td>62.722884</td>
      <td>12.736540</td>
      <td>36.021955</td>
      <td>155.552177</td>
      <td>191.574132</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.106918</td>
      <td>0.167165</td>
      <td>0.465431</td>
      <td>0.639357</td>
      <td>7.894801</td>
      <td>8.592511</td>
      <td>19.292983</td>
      <td>8.196795</td>
      <td>49.960477</td>
      <td>151.039033</td>
      <td>181.144454</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.820000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>13.940000</td>
      <td>16.665000</td>
      <td>48.000000</td>
      <td>7.001500</td>
      <td>4.000000</td>
      <td>36.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.500000</td>
      <td>24.240000</td>
      <td>63.000000</td>
      <td>12.998000</td>
      <td>17.000000</td>
      <td>118.000000</td>
      <td>145.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>27.060000</td>
      <td>31.060000</td>
      <td>78.000000</td>
      <td>16.997900</td>
      <td>49.000000</td>
      <td>222.000000</td>
      <td>284.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>41.000000</td>
      <td>50.000000</td>
      <td>100.000000</td>
      <td>56.996900</td>
      <td>367.000000</td>
      <td>886.000000</td>
      <td>977.000000</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e3bba27f-1a7c-4311-ab23-dc4b368cdcae')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e3bba27f-1a7c-4311-ab23-dc4b368cdcae button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e3bba27f-1a7c-4311-ab23-dc4b368cdcae');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-6c8d31e1-933f-4e20-a321-aba789a23f3c">
  <button class="colab-df-quickchart" onclick="quickchart('df-6c8d31e1-933f-4e20-a321-aba789a23f3c')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-6c8d31e1-933f-4e20-a321-aba789a23f3c button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>





```python
df.isnull().sum()
```




    datetime         0
    season           0
    holiday          0
    workingday       0
    weather          0
    temp             0
    atemp            0
    humidity         0
    windspeed        0
    casual        6493
    registered    6493
    count         6493
    dtype: int64




```python
df.shape
```




    (17379, 12)




```python
df = df.dropna()
```


```python
df.isnull().sum()
```




    datetime      0
    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    casual        0
    registered    0
    count         0
    dtype: int64




```python
df_corr = df.corr()
plt.figure(figsize=(14,9))
sns.heatmap(df_corr, annot=True, fmt=".2f")
plt.show()
```

    <ipython-input-10-3a9f600df7ec>:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      df_corr = df.corr()
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_10_1.png)
    



```python
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

def generate_matriz_confusion(test_labels, test_predictions_labels):
  cm = confusion_matrix(test_labels, test_predictions_labels)

  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("Rótulos Previstos")
  plt.ylabel("Rótulos Verdadeiros")
  plt.title("Matriz de Confusão")
  plt.show()
```


```python
def calcule_metrics_classification(y_test, y_pred):
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_pred)

  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-score: {f1}")
  print(f"ROC AUC: {roc_auc}")
```


```python
def plot_auc_ap(y_test, predictions):
  fpr, tpr, _ = roc_curve(y_test, predictions)
  roc_auc = roc_auc_score(y_test, predictions)

  precision, recall, _ = precision_recall_curve(y_test, predictions)
  ap = average_precision_score(y_test, predictions)

  plt.figure(figsize=(10, 5))

  # Curva ROC
  plt.subplot(1, 2, 1)
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlabel('Taxa de Falso Positivo')
  plt.ylabel('Taxa de Verdadeiro Positivo')
  plt.title('Curva ROC')
  plt.legend(loc='lower right')

  # Curva Precision-Recall
  plt.subplot(1, 2, 2)
  plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap:.2f}')
  plt.xlabel('Revocação (Recall)')
  plt.ylabel('Precisão (Precision)')
  plt.title('Curva Precision-Recall')
  plt.legend(loc='lower left')

  plt.tight_layout()
  plt.show()
```


```python
def train_and_metrics(model, X_train, y_train, X_test, y_test):
  all_predictions = []
  all_true_labels = []

  all_accuracy = []
  all_precision = []
  all_recall = []
  all_f1 = []
  all_roc_auc = []

  best_metric_value = -np.inf
  best_model = None

  cv = KFold(n_splits = 5, shuffle=True, random_state = 42)
  for n_folder, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
      X_train_fold, X_valid = X.iloc[train_index], X.iloc[test_index]
      y_train_fold, y_valid = y.iloc[train_index], y.iloc[test_index]

      model.fit(X_train_fold, y_train_fold)

      predictions = model.predict(X_valid)

      print(f'\nMatriz de confusão para o folder {n_folder}')
      generate_matriz_confusion(y_valid, predictions)

      accuracy = accuracy_score(y_valid, predictions)
      precision = precision_score(y_valid, predictions)
      recall = recall_score(y_valid, predictions)
      f1 = f1_score(y_valid, predictions)
      roc_auc = roc_auc_score(y_valid, predictions)

      print(f"Accuracy: {accuracy}")
      print(f"Precision: {precision}")
      print(f"Recall: {recall}")
      print(f"F1-score: {f1}")
      print(f"ROC AUC: {roc_auc}")

      all_accuracy.append(accuracy)
      all_precision.append(precision)
      all_recall.append(recall)
      all_f1.append(f1)
      all_roc_auc.append(roc_auc)

      if accuracy > best_metric_value:
          best_metric_value = accuracy
          best_model = model

      all_predictions.extend(predictions)
      all_true_labels.extend(y_valid)

  accuracy = accuracy_score(all_true_labels, all_predictions)
  precision = precision_score(all_true_labels, all_predictions)
  recall = recall_score(all_true_labels, all_predictions)
  f1 = f1_score(all_true_labels, all_predictions)
  roc_auc = roc_auc_score(all_true_labels, all_predictions)

  print('\n--------Métricas gerais (média) para a predição dos 5 folders--------:')
  calcule_metrics_classification(all_true_labels, all_predictions)

  predictions = rf_model.predict(X_test)
  generate_matriz_confusion(y_test, predictions)
  print()

  plot_auc_ap(y_test, predictions)
  print()

  return best_model
```

# Processamento dos dados


```python
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
```


```python
columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 'day', 'hour', 'registered', 'casual']
columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'month', 'day', 'hour', 'registered', 'casual']
```


```python
X = df[columns]
y = df['count']
```

# Classificação

> Classificando quando será alugado 150 ou mais bicicletas no dia




```python
df['count'].value_counts()
```




    5.0      169
    4.0      149
    3.0      144
    6.0      135
    2.0      132
            ... 
    801.0      1
    629.0      1
    825.0      1
    589.0      1
    636.0      1
    Name: count, Length: 822, dtype: int64




```python
max(df['count']), min(df['count'])
```




    (977.0, 1.0)




```python
plt.figure(figsize=(12, 6))
plt.plot(df['count'], marker='o', linestyle='-', color='b')
plt.title('Contagem de biciletas alugadas')
plt.xlabel('Índice da Amostra')
plt.ylabel('Count')
plt.grid(True)

plt.show()
```


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_23_0.png)
    



```python
plt.figure(figsize=(8,6))
sns.boxplot(data=df['count'], orient="v")
plt.show()
```


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_24_0.png)
    



```python
plt.figure(figsize=(8,6))
sns.boxplot(data=df[['temp','atemp']],
            orient='v')
plt.show()
```


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_25_0.png)
    



```python
limiar = 150
df['demand_class'] = (df['count'] > limiar).astype(int)

columns_classification = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'month', 'day', 'hour', 'registered', 'casual']
X = df[columns_classification]
y = df['demand_class']
```


```python
df = df.dropna()
```


```python
y.value_counts()
```




    0    5571
    1    5315
    Name: demand_class, dtype: int64




```python
X.isnull().sum()
```




    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    month         0
    day           0
    hour          0
    registered    0
    casual        0
    dtype: int64




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 456)
```


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train)
```

## Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 123)
best_rf_model = train_and_metrics(rf_model, X_train, y_train, X_test, y_test)
```

    
    Matriz de confusão para o folder 0
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_33_1.png)
    


    Accuracy: 0.9938775510204082
    Precision: 0.9944873208379272
    Recall: 0.9922992299229924
    F1-score: 0.9933920704845816
    ROC AUC: 0.9937709279967006
    
    Matriz de confusão para o folder 1
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_33_3.png)
    


    Accuracy: 0.9908163265306122
    Precision: 0.9905163329820864
    Recall: 0.9905163329820864
    F1-score: 0.9905163329820864
    ROC AUC: 0.9908071279153755
    
    Matriz de confusão para o folder 2
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_33_5.png)
    


    Accuracy: 0.9964267483409903
    Precision: 0.9966480446927374
    Recall: 0.9955357142857143
    F1-score: 0.9960915689558906
    ROC AUC: 0.9963567564843436
    
    Matriz de confusão para o folder 3
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_33_7.png)
    


    Accuracy: 0.9948953547728433
    Precision: 0.9948132780082988
    Recall: 0.9948132780082988
    F1-score: 0.9948132780082988
    ROC AUC: 0.9948940761900791
    
    Matriz de confusão para o folder 4
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_33_9.png)
    


    Accuracy: 0.9943848902501277
    Precision: 0.9915700737618546
    Recall: 0.996822033898305
    F1-score: 0.99418911780243
    ROC AUC: 0.9944701302496451
    
    --------Métricas gerais (média) para a predição dos 5 folders--------:
    Accuracy: 0.9940798203531693
    Precision: 0.9935677530017153
    Recall: 0.993993993993994
    F1-score: 0.9937808277932663
    ROC AUC: 0.994075867493589
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_33_11.png)
    


    
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_33_13.png)
    


    
    

## SVM


```python
from sklearn.svm import SVC
svc_model = SVC(kernel='linear', C = 1.0, random_state = 548)

best_svc_model = train_and_metrics(svc_model, X_train, y_train, X_test, y_test)
```

    
    Matriz de confusão para o folder 0
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_35_1.png)
    


    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1-score: 1.0
    ROC AUC: 1.0
    
    Matriz de confusão para o folder 1
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_35_3.png)
    


    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1-score: 1.0
    ROC AUC: 1.0
    
    Matriz de confusão para o folder 2
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_35_5.png)
    


    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1-score: 1.0
    ROC AUC: 1.0
    
    Matriz de confusão para o folder 3
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_35_7.png)
    


    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1-score: 1.0
    ROC AUC: 1.0
    
    Matriz de confusão para o folder 4
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_35_9.png)
    


    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1-score: 1.0
    ROC AUC: 1.0
    
    --------Métricas gerais (média) para a predição dos 5 folders--------:
    Accuracy: 1.0
    Precision: 1.0
    Recall: 1.0
    F1-score: 1.0
    ROC AUC: 1.0
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_35_11.png)
    


    
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_35_13.png)
    


    
    

## Rede Neural

### Rede Bayesiana


```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```


```python
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_normalized = scaler.fit_transform(X_train)
```


```python
from sklearn.naive_bayes import MultinomialNB
bayes_model = MultinomialNB()

best_bayes_model = train_and_metrics(bayes_model, X_train, y_train, X_test, y_test)
```

    
    Matriz de confusão para o folder 0
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_40_1.png)
    


    Accuracy: 0.938265306122449
    Precision: 0.9012219959266803
    Recall: 0.9735973597359736
    F1-score: 0.9360126916975144
    ROC AUC: 0.9406521527509555
    
    Matriz de confusão para o folder 1
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_40_3.png)
    


    Accuracy: 0.9448979591836735
    Precision: 0.9151036525172754
    Recall: 0.9768177028451
    F1-score: 0.944954128440367
    ROC AUC: 0.9458767050328368
    
    Matriz de confusão para o folder 2
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_40_5.png)
    


    Accuracy: 0.9530372639101582
    Precision: 0.9161490683229814
    Recall: 0.9877232142857143
    F1-score: 0.9505907626208377
    ROC AUC: 0.9557618893629889
    
    Matriz de confusão para o folder 3
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_40_7.png)
    


    Accuracy: 0.9469116896375702
    Precision: 0.9240631163708086
    Recall: 0.9719917012448133
    F1-score: 0.9474216380182002
    ROC AUC: 0.9473023832857232
    
    Matriz de confusão para o folder 4
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_40_9.png)
    


    Accuracy: 0.9428279734558448
    Precision: 0.9126984126984127
    Recall: 0.9745762711864406
    F1-score: 0.9426229508196721
    ROC AUC: 0.9439383818986391
    
    --------Métricas gerais (média) para a predição dos 5 folders--------:
    Accuracy: 0.9451873022353782
    Precision: 0.9139072847682119
    Recall: 0.9768339768339769
    F1-score: 0.9443234836702956
    ROC AUC: 0.9466448365182543
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_40_11.png)
    


    
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_40_13.png)
    


    
    


```python
# scaler = StandardScaler()
# X_train_normalized = scaler.fit_transform(X_train)
```


```python
# X_train_normalized.shape
```


```python
y_train
```




    10805    1
    133      0
    3815     0
    6739     0
    10507    0
            ..
    9116     0
    10735    1
    6186     0
    10841    0
    613      1
    Name: demand_class, Length: 9797, dtype: int64



### Convolucional


```python
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```


```python
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_train.shape
```




    (9797, 13, 1)




```python
name_checkpoint_best_model = 'best_model.h5'
```


```python
X_train.shape, y_train.shape
```




    ((9797, 13), (9797,))




```python
!pip install tensorflow_addons

import tensorflow_addons as tfa
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true)))

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # soma dos verdadeiros positivos
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) # soma dos verdadeiros positivos + falsos negativos
    recall = true_positives / (possible_positives + K.epsilon()) # cálculo do recall
    return recall

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1))) # soma dos verdadeiros negativos
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1))) # soma dos verdadeiros negativos + falsos positivos
    specificity = true_negatives / (possible_negatives + K.epsilon()) # cálculo da especificidade
    return specificity
```

    Requirement already satisfied: tensorflow_addons in /usr/local/lib/python3.10/dist-packages (0.22.0)
    Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow_addons) (23.2)
    Requirement already satisfied: typeguard<3.0.0,>=2.7 in /usr/local/lib/python3.10/dist-packages (from tensorflow_addons) (2.13.3)
    


```python
X_train.shape
```




    (9797, 13)




```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state = 789)
accuracies = []

# for train_index, test_index in cv.split(X_train, y_train):
#     X_train_neural, X_test_neural = X_train[train_index], X_train[test_index]
#     y_train_neural, y_test_neural = y_train.iloc[train_index], y_train.iloc[test_index]
for n_folder, (train_index, test_index) in enumerate(cv.split(X_train, y_train)):
    X_train_neural, X_test_neural = X_train[train_index], X_train[test_index]
    y_train_neural, y_test_neural = y_train.iloc[train_index], y_train.iloc[test_index]

    model_checkpoint = ModelCheckpoint(name_checkpoint_best_model, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2], )),
        Conv1D(64, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics = [ tf.keras.metrics.MeanAbsoluteError(name='mae'),
                                  tf.keras.metrics.CategoricalAccuracy(name='accuracy', dtype=None),
                                  tf.keras.metrics.Recall(name='Recall'),
                                  tf.keras.metrics.Precision(name='Precision'),
                                  tf.keras.metrics.SpecificityAtSensitivity(name='Specificity', sensitivity = 0.5),
                                  tf.keras.metrics.AUC(curve="ROC", summation_method="interpolation", name="AUC")
                                  ])

    model.fit(X_train_neural, y_train_neural, epochs = 20, batch_size = 32,
              validation_data=(X_test_neural, y_test_neural), callbacks=[early_stopping, model_checkpoint], verbose=1 )
    my_tensor = tf.convert_to_tensor(X_test_neural, dtype=tf.float16)
    previsoes = model.predict(my_tensor)

    y_pred_classes = (previsoes > 0.9)
    y_val_classes = np.round(y_test_neural)
    generate_matriz_confusion(y_val_classes, y_pred_classes)
    calcule_metrics_classification(y_val_classes, y_pred_classes)
```

    Epoch 1/20
    245/245 [==============================] - ETA: 0s - loss: 0.2841 - mae: 0.2060 - accuracy: 1.0000 - Recall: 0.8786 - Precision: 0.8791 - Specificity: 0.9965 - AUC: 0.9563
    Epoch 1: val_accuracy improved from -inf to 1.00000, saving model to best_model.h5
    245/245 [==============================] - 8s 16ms/step - loss: 0.2841 - mae: 0.2060 - accuracy: 1.0000 - Recall: 0.8786 - Precision: 0.8791 - Specificity: 0.9965 - AUC: 0.9563 - val_loss: 0.0953 - val_mae: 0.0755 - val_accuracy: 1.0000 - val_Recall: 0.9885 - val_Precision: 0.9518 - val_Specificity: 1.0000 - val_AUC: 0.9983
    Epoch 2/20
      8/245 [..............................] - ETA: 1s - loss: 0.1225 - mae: 0.0828 - accuracy: 1.0000 - Recall: 0.9615 - Precision: 0.9091 - Specificity: 1.0000 - AUC: 0.9923

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    

    242/245 [============================>.] - ETA: 0s - loss: 0.0863 - mae: 0.0621 - accuracy: 1.0000 - Recall: 0.9640 - Precision: 0.9645 - Specificity: 1.0000 - AUC: 0.9962
    Epoch 2: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0859 - mae: 0.0619 - accuracy: 1.0000 - Recall: 0.9645 - Precision: 0.9650 - Specificity: 1.0000 - AUC: 0.9962 - val_loss: 0.0556 - val_mae: 0.0432 - val_accuracy: 1.0000 - val_Recall: 0.9697 - val_Precision: 0.9968 - val_Specificity: 1.0000 - val_AUC: 0.9993
    Epoch 3/20
    242/245 [============================>.] - ETA: 0s - loss: 0.0601 - mae: 0.0414 - accuracy: 1.0000 - Recall: 0.9762 - Precision: 0.9770 - Specificity: 1.0000 - AUC: 0.9980
    Epoch 3: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0606 - mae: 0.0417 - accuracy: 1.0000 - Recall: 0.9765 - Precision: 0.9757 - Specificity: 1.0000 - AUC: 0.9980 - val_loss: 0.0405 - val_mae: 0.0324 - val_accuracy: 1.0000 - val_Recall: 0.9833 - val_Precision: 0.9937 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 4/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0427 - mae: 0.0317 - accuracy: 1.0000 - Recall: 0.9832 - Precision: 0.9838 - Specificity: 1.0000 - AUC: 0.9992
    Epoch 4: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0430 - mae: 0.0318 - accuracy: 1.0000 - Recall: 0.9825 - Precision: 0.9841 - Specificity: 1.0000 - AUC: 0.9991 - val_loss: 0.0312 - val_mae: 0.0248 - val_accuracy: 1.0000 - val_Recall: 0.9906 - val_Precision: 0.9896 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 5/20
    237/245 [============================>.] - ETA: 0s - loss: 0.0414 - mae: 0.0287 - accuracy: 1.0000 - Recall: 0.9806 - Precision: 0.9806 - Specificity: 1.0000 - AUC: 0.9990
    Epoch 5: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0420 - mae: 0.0290 - accuracy: 1.0000 - Recall: 0.9796 - Precision: 0.9807 - Specificity: 1.0000 - AUC: 0.9990 - val_loss: 0.0588 - val_mae: 0.0341 - val_accuracy: 1.0000 - val_Recall: 0.9384 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 6/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0424 - mae: 0.0271 - accuracy: 1.0000 - Recall: 0.9824 - Precision: 0.9790 - Specificity: 1.0000 - AUC: 0.9989
    Epoch 6: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0421 - mae: 0.0270 - accuracy: 1.0000 - Recall: 0.9825 - Precision: 0.9794 - Specificity: 1.0000 - AUC: 0.9989 - val_loss: 0.0295 - val_mae: 0.0219 - val_accuracy: 1.0000 - val_Recall: 0.9979 - val_Precision: 0.9795 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 7/20
    232/245 [===========================>..] - ETA: 0s - loss: 0.0394 - mae: 0.0250 - accuracy: 1.0000 - Recall: 0.9815 - Precision: 0.9796 - Specificity: 1.0000 - AUC: 0.9991
    Epoch 7: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0392 - mae: 0.0251 - accuracy: 1.0000 - Recall: 0.9815 - Precision: 0.9799 - Specificity: 1.0000 - AUC: 0.9991 - val_loss: 0.0243 - val_mae: 0.0188 - val_accuracy: 1.0000 - val_Recall: 0.9896 - val_Precision: 0.9958 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 8/20
    239/245 [============================>.] - ETA: 0s - loss: 0.0327 - mae: 0.0212 - accuracy: 1.0000 - Recall: 0.9859 - Precision: 0.9864 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 8: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0325 - mae: 0.0211 - accuracy: 1.0000 - Recall: 0.9862 - Precision: 0.9864 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0261 - val_mae: 0.0185 - val_accuracy: 1.0000 - val_Recall: 0.9823 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 9/20
    236/245 [===========================>..] - ETA: 0s - loss: 0.0278 - mae: 0.0187 - accuracy: 1.0000 - Recall: 0.9875 - Precision: 0.9897 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 9: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0271 - mae: 0.0183 - accuracy: 1.0000 - Recall: 0.9880 - Precision: 0.9901 - Specificity: 1.0000 - AUC: 0.9996 - val_loss: 0.0263 - val_mae: 0.0188 - val_accuracy: 1.0000 - val_Recall: 0.9781 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 10/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0355 - mae: 0.0207 - accuracy: 1.0000 - Recall: 0.9833 - Precision: 0.9822 - Specificity: 0.9997 - AUC: 0.9991
    Epoch 10: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0360 - mae: 0.0209 - accuracy: 1.0000 - Recall: 0.9825 - Precision: 0.9825 - Specificity: 0.9998 - AUC: 0.9991 - val_loss: 0.0342 - val_mae: 0.0216 - val_accuracy: 1.0000 - val_Recall: 0.9729 - val_Precision: 0.9957 - val_Specificity: 1.0000 - val_AUC: 0.9996
    Epoch 11/20
    238/245 [============================>.] - ETA: 0s - loss: 0.0484 - mae: 0.0255 - accuracy: 1.0000 - Recall: 0.9817 - Precision: 0.9781 - Specificity: 0.9995 - AUC: 0.9984
    Epoch 11: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 6ms/step - loss: 0.0487 - mae: 0.0255 - accuracy: 1.0000 - Recall: 0.9809 - Precision: 0.9784 - Specificity: 0.9995 - AUC: 0.9984 - val_loss: 0.0235 - val_mae: 0.0179 - val_accuracy: 1.0000 - val_Recall: 0.9875 - val_Precision: 0.9927 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 12/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0278 - mae: 0.0177 - accuracy: 1.0000 - Recall: 0.9875 - Precision: 0.9875 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 12: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0282 - mae: 0.0179 - accuracy: 1.0000 - Recall: 0.9877 - Precision: 0.9870 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0222 - val_mae: 0.0158 - val_accuracy: 1.0000 - val_Recall: 0.9969 - val_Precision: 0.9845 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 13/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0350 - mae: 0.0202 - accuracy: 1.0000 - Recall: 0.9850 - Precision: 0.9853 - Specificity: 1.0000 - AUC: 0.9993
    Epoch 13: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0352 - mae: 0.0203 - accuracy: 1.0000 - Recall: 0.9851 - Precision: 0.9849 - Specificity: 1.0000 - AUC: 0.9993 - val_loss: 0.0222 - val_mae: 0.0160 - val_accuracy: 1.0000 - val_Recall: 0.9958 - val_Precision: 0.9907 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 14/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0276 - mae: 0.0176 - accuracy: 1.0000 - Recall: 0.9878 - Precision: 0.9883 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 14: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0276 - mae: 0.0176 - accuracy: 1.0000 - Recall: 0.9880 - Precision: 0.9877 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0169 - val_mae: 0.0135 - val_accuracy: 1.0000 - val_Recall: 0.9958 - val_Precision: 0.9927 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 15/20
    236/245 [===========================>..] - ETA: 0s - loss: 0.0262 - mae: 0.0159 - accuracy: 1.0000 - Recall: 0.9886 - Precision: 0.9900 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 15: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0259 - mae: 0.0158 - accuracy: 1.0000 - Recall: 0.9885 - Precision: 0.9901 - Specificity: 1.0000 - AUC: 0.9996 - val_loss: 0.0228 - val_mae: 0.0157 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9785 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 16/20
    239/245 [============================>.] - ETA: 0s - loss: 0.0311 - mae: 0.0172 - accuracy: 1.0000 - Recall: 0.9875 - Precision: 0.9875 - Specificity: 0.9997 - AUC: 0.9989
    Epoch 16: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 8ms/step - loss: 0.0308 - mae: 0.0171 - accuracy: 1.0000 - Recall: 0.9872 - Precision: 0.9877 - Specificity: 0.9998 - AUC: 0.9990 - val_loss: 0.0200 - val_mae: 0.0143 - val_accuracy: 1.0000 - val_Recall: 0.9854 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 17/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0234 - mae: 0.0151 - accuracy: 1.0000 - Recall: 0.9902 - Precision: 0.9894 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 17: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 10ms/step - loss: 0.0239 - mae: 0.0153 - accuracy: 1.0000 - Recall: 0.9901 - Precision: 0.9890 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.1373 - val_mae: 0.0448 - val_accuracy: 1.0000 - val_Recall: 0.9081 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9946
    Epoch 18/20
    230/245 [===========================>..] - ETA: 0s - loss: 0.0496 - mae: 0.0244 - accuracy: 1.0000 - Recall: 0.9791 - Precision: 0.9811 - Specificity: 0.9997 - AUC: 0.9977
    Epoch 18: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0476 - mae: 0.0238 - accuracy: 1.0000 - Recall: 0.9802 - Precision: 0.9814 - Specificity: 0.9998 - AUC: 0.9979 - val_loss: 0.0196 - val_mae: 0.0147 - val_accuracy: 1.0000 - val_Recall: 0.9948 - val_Precision: 0.9948 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 19/20
    235/245 [===========================>..] - ETA: 0s - loss: 0.0234 - mae: 0.0154 - accuracy: 1.0000 - Recall: 0.9885 - Precision: 0.9899 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 19: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0234 - mae: 0.0154 - accuracy: 1.0000 - Recall: 0.9885 - Precision: 0.9898 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0250 - val_mae: 0.0158 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9755 - val_Specificity: 1.0000 - val_AUC: 0.9999
    62/62 [==============================] - 0s 2ms/step
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_51_3.png)
    


    Accuracy: 0.9811224489795919
    Precision: 1.0
    Recall: 0.9613778705636743
    F1-score: 0.9803086748270357
    ROC AUC: 0.9806889352818371
    Epoch 1/20
    238/245 [============================>.] - ETA: 0s - loss: 0.2880 - mae: 0.2109 - accuracy: 1.0000 - Recall: 0.8470 - Precision: 0.8971 - Specificity: 0.9951 - AUC: 0.9568
    Epoch 1: val_accuracy improved from -inf to 1.00000, saving model to best_model.h5
    245/245 [==============================] - 4s 7ms/step - loss: 0.2827 - mae: 0.2072 - accuracy: 1.0000 - Recall: 0.8499 - Precision: 0.8994 - Specificity: 0.9965 - AUC: 0.9586 - val_loss: 0.1013 - val_mae: 0.0793 - val_accuracy: 1.0000 - val_Recall: 0.9468 - val_Precision: 0.9902 - val_Specificity: 1.0000 - val_AUC: 0.9976
    Epoch 2/20
     46/245 [====>.........................] - ETA: 0s - loss: 0.0929 - mae: 0.0734 - accuracy: 1.0000 - Recall: 0.9577 - Precision: 0.9776 - Specificity: 1.0000 - AUC: 0.9974

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    

    238/245 [============================>.] - ETA: 0s - loss: 0.0861 - mae: 0.0633 - accuracy: 1.0000 - Recall: 0.9653 - Precision: 0.9656 - Specificity: 1.0000 - AUC: 0.9965
    Epoch 2: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0854 - mae: 0.0628 - accuracy: 1.0000 - Recall: 0.9663 - Precision: 0.9658 - Specificity: 1.0000 - AUC: 0.9965 - val_loss: 0.0594 - val_mae: 0.0461 - val_accuracy: 1.0000 - val_Recall: 0.9760 - val_Precision: 0.9926 - val_Specificity: 1.0000 - val_AUC: 0.9990
    Epoch 3/20
    232/245 [===========================>..] - ETA: 0s - loss: 0.0571 - mae: 0.0420 - accuracy: 1.0000 - Recall: 0.9758 - Precision: 0.9750 - Specificity: 1.0000 - AUC: 0.9984
    Epoch 3: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0560 - mae: 0.0413 - accuracy: 1.0000 - Recall: 0.9760 - Precision: 0.9755 - Specificity: 1.0000 - AUC: 0.9985 - val_loss: 0.0466 - val_mae: 0.0351 - val_accuracy: 1.0000 - val_Recall: 0.9916 - val_Precision: 0.9714 - val_Specificity: 1.0000 - val_AUC: 0.9993
    Epoch 4/20
    234/245 [===========================>..] - ETA: 0s - loss: 0.0568 - mae: 0.0362 - accuracy: 1.0000 - Recall: 0.9751 - Precision: 0.9775 - Specificity: 0.9997 - AUC: 0.9980
    Epoch 4: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0569 - mae: 0.0363 - accuracy: 1.0000 - Recall: 0.9757 - Precision: 0.9773 - Specificity: 0.9998 - AUC: 0.9980 - val_loss: 0.0393 - val_mae: 0.0304 - val_accuracy: 1.0000 - val_Recall: 0.9739 - val_Precision: 0.9979 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 5/20
    239/245 [============================>.] - ETA: 0s - loss: 0.0437 - mae: 0.0299 - accuracy: 1.0000 - Recall: 0.9826 - Precision: 0.9802 - Specificity: 1.0000 - AUC: 0.9989
    Epoch 5: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0440 - mae: 0.0300 - accuracy: 1.0000 - Recall: 0.9823 - Precision: 0.9805 - Specificity: 1.0000 - AUC: 0.9989 - val_loss: 0.0418 - val_mae: 0.0288 - val_accuracy: 1.0000 - val_Recall: 0.9958 - val_Precision: 0.9666 - val_Specificity: 1.0000 - val_AUC: 0.9996
    Epoch 6/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0367 - mae: 0.0256 - accuracy: 1.0000 - Recall: 0.9842 - Precision: 0.9835 - Specificity: 1.0000 - AUC: 0.9993
    Epoch 6: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0367 - mae: 0.0257 - accuracy: 1.0000 - Recall: 0.9841 - Precision: 0.9836 - Specificity: 1.0000 - AUC: 0.9993 - val_loss: 0.0310 - val_mae: 0.0229 - val_accuracy: 1.0000 - val_Recall: 0.9749 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 7/20
    229/245 [===========================>..] - ETA: 0s - loss: 0.0410 - mae: 0.0254 - accuracy: 1.0000 - Recall: 0.9812 - Precision: 0.9826 - Specificity: 1.0000 - AUC: 0.9990
    Epoch 7: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0408 - mae: 0.0253 - accuracy: 1.0000 - Recall: 0.9807 - Precision: 0.9827 - Specificity: 1.0000 - AUC: 0.9990 - val_loss: 0.0298 - val_mae: 0.0217 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9766 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 8/20
    235/245 [===========================>..] - ETA: 0s - loss: 0.0307 - mae: 0.0212 - accuracy: 1.0000 - Recall: 0.9858 - Precision: 0.9880 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 8: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0300 - mae: 0.0208 - accuracy: 1.0000 - Recall: 0.9864 - Precision: 0.9882 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0222 - val_mae: 0.0177 - val_accuracy: 1.0000 - val_Recall: 0.9885 - val_Precision: 0.9979 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 9/20
    234/245 [===========================>..] - ETA: 0s - loss: 0.0284 - mae: 0.0192 - accuracy: 1.0000 - Recall: 0.9872 - Precision: 0.9875 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 9: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0279 - mae: 0.0189 - accuracy: 1.0000 - Recall: 0.9875 - Precision: 0.9875 - Specificity: 1.0000 - AUC: 0.9996 - val_loss: 0.0275 - val_mae: 0.0194 - val_accuracy: 1.0000 - val_Recall: 0.9739 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 10/20
    233/245 [===========================>..] - ETA: 0s - loss: 0.0270 - mae: 0.0177 - accuracy: 1.0000 - Recall: 0.9868 - Precision: 0.9890 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 10: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0275 - mae: 0.0181 - accuracy: 1.0000 - Recall: 0.9867 - Precision: 0.9890 - Specificity: 1.0000 - AUC: 0.9996 - val_loss: 0.0299 - val_mae: 0.0193 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9716 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 11/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0340 - mae: 0.0200 - accuracy: 1.0000 - Recall: 0.9837 - Precision: 0.9847 - Specificity: 1.0000 - AUC: 0.9993
    Epoch 11: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0339 - mae: 0.0199 - accuracy: 1.0000 - Recall: 0.9838 - Precision: 0.9846 - Specificity: 1.0000 - AUC: 0.9993 - val_loss: 0.0236 - val_mae: 0.0159 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9816 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 12/20
    238/245 [============================>.] - ETA: 0s - loss: 0.0231 - mae: 0.0162 - accuracy: 1.0000 - Recall: 0.9890 - Precision: 0.9909 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 12: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0239 - mae: 0.0164 - accuracy: 1.0000 - Recall: 0.9888 - Precision: 0.9908 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.1015 - val_mae: 0.0398 - val_accuracy: 1.0000 - val_Recall: 0.9175 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 13/20
    242/245 [============================>.] - ETA: 0s - loss: 0.0367 - mae: 0.0204 - accuracy: 1.0000 - Recall: 0.9849 - Precision: 0.9834 - Specificity: 1.0000 - AUC: 0.9992
    Epoch 13: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0365 - mae: 0.0202 - accuracy: 1.0000 - Recall: 0.9851 - Precision: 0.9836 - Specificity: 1.0000 - AUC: 0.9992 - val_loss: 0.0186 - val_mae: 0.0142 - val_accuracy: 1.0000 - val_Recall: 0.9958 - val_Precision: 0.9937 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 14/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0271 - mae: 0.0164 - accuracy: 1.0000 - Recall: 0.9897 - Precision: 0.9882 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 14: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0283 - mae: 0.0167 - accuracy: 1.0000 - Recall: 0.9890 - Precision: 0.9883 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0462 - val_mae: 0.0244 - val_accuracy: 1.0000 - val_Recall: 0.9582 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 15/20
    242/245 [============================>.] - ETA: 0s - loss: 0.0336 - mae: 0.0188 - accuracy: 1.0000 - Recall: 0.9847 - Precision: 0.9865 - Specificity: 1.0000 - AUC: 0.9993
    Epoch 15: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0335 - mae: 0.0187 - accuracy: 1.0000 - Recall: 0.9846 - Precision: 0.9867 - Specificity: 1.0000 - AUC: 0.9993 - val_loss: 0.0192 - val_mae: 0.0144 - val_accuracy: 1.0000 - val_Recall: 0.9937 - val_Precision: 0.9958 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 16/20
    235/245 [===========================>..] - ETA: 0s - loss: 0.0246 - mae: 0.0160 - accuracy: 1.0000 - Recall: 0.9900 - Precision: 0.9894 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 16: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0249 - mae: 0.0162 - accuracy: 1.0000 - Recall: 0.9896 - Precision: 0.9896 - Specificity: 1.0000 - AUC: 0.9996 - val_loss: 0.0182 - val_mae: 0.0133 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9876 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 17/20
    235/245 [===========================>..] - ETA: 0s - loss: 0.0238 - mae: 0.0154 - accuracy: 1.0000 - Recall: 0.9894 - Precision: 0.9891 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 17: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0235 - mae: 0.0153 - accuracy: 1.0000 - Recall: 0.9896 - Precision: 0.9893 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0179 - val_mae: 0.0128 - val_accuracy: 1.0000 - val_Recall: 0.9948 - val_Precision: 0.9917 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 18/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0276 - mae: 0.0166 - accuracy: 1.0000 - Recall: 0.9872 - Precision: 0.9872 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 18: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0279 - mae: 0.0167 - accuracy: 1.0000 - Recall: 0.9872 - Precision: 0.9870 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0207 - val_mae: 0.0136 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9806 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 19/20
    232/245 [===========================>..] - ETA: 0s - loss: 0.0346 - mae: 0.0181 - accuracy: 1.0000 - Recall: 0.9860 - Precision: 0.9857 - Specificity: 0.9997 - AUC: 0.9991
    Epoch 19: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0343 - mae: 0.0181 - accuracy: 1.0000 - Recall: 0.9859 - Precision: 0.9859 - Specificity: 0.9998 - AUC: 0.9991 - val_loss: 0.0151 - val_mae: 0.0120 - val_accuracy: 1.0000 - val_Recall: 0.9927 - val_Precision: 0.9958 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 20/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0210 - mae: 0.0140 - accuracy: 1.0000 - Recall: 0.9894 - Precision: 0.9918 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 20: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0209 - mae: 0.0140 - accuracy: 1.0000 - Recall: 0.9896 - Precision: 0.9919 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0170 - val_mae: 0.0123 - val_accuracy: 1.0000 - val_Recall: 0.9958 - val_Precision: 0.9876 - val_Specificity: 1.0000 - val_AUC: 0.9999
    62/62 [==============================] - 0s 2ms/step
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_51_7.png)
    


    Accuracy: 0.986734693877551
    Precision: 1.0
    Recall: 0.9728601252609603
    F1-score: 0.9862433862433863
    ROC AUC: 0.9864300626304802
    Epoch 1/20
    238/245 [============================>.] - ETA: 0s - loss: 0.2884 - mae: 0.2106 - accuracy: 1.0000 - Recall: 0.9108 - Precision: 0.8631 - Specificity: 0.9956 - AUC: 0.9582
    Epoch 1: val_accuracy improved from -inf to 1.00000, saving model to best_model.h5
    245/245 [==============================] - 3s 6ms/step - loss: 0.2834 - mae: 0.2071 - accuracy: 1.0000 - Recall: 0.9136 - Precision: 0.8655 - Specificity: 0.9968 - AUC: 0.9598 - val_loss: 0.1080 - val_mae: 0.0848 - val_accuracy: 1.0000 - val_Recall: 0.9781 - val_Precision: 0.9484 - val_Specificity: 1.0000 - val_AUC: 0.9967
    Epoch 2/20
     43/245 [====>.........................] - ETA: 0s - loss: 0.1067 - mae: 0.0818 - accuracy: 1.0000 - Recall: 0.9555 - Precision: 0.9643 - Specificity: 1.0000 - AUC: 0.9956

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    

    233/245 [===========================>..] - ETA: 0s - loss: 0.0794 - mae: 0.0604 - accuracy: 1.0000 - Recall: 0.9733 - Precision: 0.9736 - Specificity: 1.0000 - AUC: 0.9975
    Epoch 2: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0781 - mae: 0.0595 - accuracy: 1.0000 - Recall: 0.9734 - Precision: 0.9744 - Specificity: 1.0000 - AUC: 0.9976 - val_loss: 0.0621 - val_mae: 0.0477 - val_accuracy: 1.0000 - val_Recall: 0.9603 - val_Precision: 0.9924 - val_Specificity: 1.0000 - val_AUC: 0.9989
    Epoch 3/20
    236/245 [===========================>..] - ETA: 0s - loss: 0.0500 - mae: 0.0377 - accuracy: 1.0000 - Recall: 0.9816 - Precision: 0.9808 - Specificity: 1.0000 - AUC: 0.9989
    Epoch 3: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0498 - mae: 0.0375 - accuracy: 1.0000 - Recall: 0.9817 - Precision: 0.9810 - Specificity: 1.0000 - AUC: 0.9990 - val_loss: 0.0420 - val_mae: 0.0329 - val_accuracy: 1.0000 - val_Recall: 0.9708 - val_Precision: 0.9968 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 4/20
    238/245 [============================>.] - ETA: 0s - loss: 0.0554 - mae: 0.0359 - accuracy: 1.0000 - Recall: 0.9755 - Precision: 0.9774 - Specificity: 1.0000 - AUC: 0.9982
    Epoch 4: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0556 - mae: 0.0359 - accuracy: 1.0000 - Recall: 0.9757 - Precision: 0.9770 - Specificity: 1.0000 - AUC: 0.9981 - val_loss: 0.0476 - val_mae: 0.0343 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9599 - val_Specificity: 1.0000 - val_AUC: 0.9995
    Epoch 5/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0366 - mae: 0.0266 - accuracy: 1.0000 - Recall: 0.9856 - Precision: 0.9861 - Specificity: 1.0000 - AUC: 0.9993
    Epoch 5: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0368 - mae: 0.0267 - accuracy: 1.0000 - Recall: 0.9854 - Precision: 0.9862 - Specificity: 1.0000 - AUC: 0.9993 - val_loss: 0.0353 - val_mae: 0.0264 - val_accuracy: 1.0000 - val_Recall: 0.9718 - val_Precision: 0.9957 - val_Specificity: 1.0000 - val_AUC: 0.9996
    Epoch 6/20
    232/245 [===========================>..] - ETA: 0s - loss: 0.0331 - mae: 0.0236 - accuracy: 1.0000 - Recall: 0.9864 - Precision: 0.9875 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 6: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0331 - mae: 0.0236 - accuracy: 1.0000 - Recall: 0.9864 - Precision: 0.9875 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0294 - val_mae: 0.0226 - val_accuracy: 1.0000 - val_Recall: 0.9969 - val_Precision: 0.9835 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 7/20
    244/245 [============================>.] - ETA: 0s - loss: 0.0327 - mae: 0.0215 - accuracy: 1.0000 - Recall: 0.9866 - Precision: 0.9848 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 7: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0327 - mae: 0.0215 - accuracy: 1.0000 - Recall: 0.9867 - Precision: 0.9849 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0262 - val_mae: 0.0200 - val_accuracy: 1.0000 - val_Recall: 0.9791 - val_Precision: 0.9979 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 8/20
    233/245 [===========================>..] - ETA: 0s - loss: 0.0501 - mae: 0.0252 - accuracy: 1.0000 - Recall: 0.9816 - Precision: 0.9808 - Specificity: 0.9992 - AUC: 0.9979
    Epoch 8: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0526 - mae: 0.0259 - accuracy: 1.0000 - Recall: 0.9802 - Precision: 0.9807 - Specificity: 0.9993 - AUC: 0.9978 - val_loss: 0.0984 - val_mae: 0.0444 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9238 - val_Specificity: 1.0000 - val_AUC: 0.9992
    Epoch 9/20
    235/245 [===========================>..] - ETA: 0s - loss: 0.0292 - mae: 0.0202 - accuracy: 1.0000 - Recall: 0.9896 - Precision: 0.9875 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 9: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 6ms/step - loss: 0.0290 - mae: 0.0202 - accuracy: 1.0000 - Recall: 0.9896 - Precision: 0.9880 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0250 - val_mae: 0.0192 - val_accuracy: 1.0000 - val_Recall: 0.9948 - val_Precision: 0.9865 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 10/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0221 - mae: 0.0159 - accuracy: 1.0000 - Recall: 0.9924 - Precision: 0.9913 - Specificity: 1.0000 - AUC: 0.9998
    Epoch 10: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 6ms/step - loss: 0.0222 - mae: 0.0160 - accuracy: 1.0000 - Recall: 0.9924 - Precision: 0.9909 - Specificity: 1.0000 - AUC: 0.9998 - val_loss: 0.0551 - val_mae: 0.0304 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9457 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 11/20
    242/245 [============================>.] - ETA: 0s - loss: 0.0276 - mae: 0.0177 - accuracy: 1.0000 - Recall: 0.9889 - Precision: 0.9879 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 11: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 7ms/step - loss: 0.0274 - mae: 0.0176 - accuracy: 1.0000 - Recall: 0.9888 - Precision: 0.9880 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0210 - val_mae: 0.0162 - val_accuracy: 1.0000 - val_Recall: 0.9969 - val_Precision: 0.9907 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 12/20
    236/245 [===========================>..] - ETA: 0s - loss: 0.0233 - mae: 0.0149 - accuracy: 1.0000 - Recall: 0.9919 - Precision: 0.9908 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 12: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0232 - mae: 0.0149 - accuracy: 1.0000 - Recall: 0.9919 - Precision: 0.9906 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0218 - val_mae: 0.0166 - val_accuracy: 1.0000 - val_Recall: 0.9885 - val_Precision: 0.9916 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 13/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0372 - mae: 0.0192 - accuracy: 1.0000 - Recall: 0.9859 - Precision: 0.9851 - Specificity: 0.9995 - AUC: 0.9990
    Epoch 13: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0369 - mae: 0.0191 - accuracy: 1.0000 - Recall: 0.9856 - Precision: 0.9854 - Specificity: 0.9995 - AUC: 0.9990 - val_loss: 0.0377 - val_mae: 0.0208 - val_accuracy: 1.0000 - val_Recall: 0.9624 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 14/20
    233/245 [===========================>..] - ETA: 0s - loss: 0.0214 - mae: 0.0145 - accuracy: 1.0000 - Recall: 0.9920 - Precision: 0.9901 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 14: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0229 - mae: 0.0151 - accuracy: 1.0000 - Recall: 0.9911 - Precision: 0.9893 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0415 - val_mae: 0.0235 - val_accuracy: 1.0000 - val_Recall: 0.9979 - val_Precision: 0.9618 - val_Specificity: 1.0000 - val_AUC: 0.9996
    Epoch 15/20
    244/245 [============================>.] - ETA: 0s - loss: 0.0313 - mae: 0.0177 - accuracy: 1.0000 - Recall: 0.9869 - Precision: 0.9861 - Specificity: 1.0000 - AUC: 0.9993
    Epoch 15: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0312 - mae: 0.0177 - accuracy: 1.0000 - Recall: 0.9869 - Precision: 0.9862 - Specificity: 1.0000 - AUC: 0.9993 - val_loss: 0.0498 - val_mae: 0.0266 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9523 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 16/20
    234/245 [===========================>..] - ETA: 0s - loss: 0.0218 - mae: 0.0139 - accuracy: 1.0000 - Recall: 0.9899 - Precision: 0.9913 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 16: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0215 - mae: 0.0138 - accuracy: 1.0000 - Recall: 0.9903 - Precision: 0.9914 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0257 - val_mae: 0.0164 - val_accuracy: 1.0000 - val_Recall: 0.9823 - val_Precision: 0.9947 - val_Specificity: 1.0000 - val_AUC: 0.9997
    62/62 [==============================] - 0s 2ms/step
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_51_11.png)
    


    Accuracy: 0.9800918836140888
    Precision: 1.0
    Recall: 0.9592901878914405
    F1-score: 0.9792221630261055
    ROC AUC: 0.9796450939457202
    Epoch 1/20
    241/245 [============================>.] - ETA: 0s - loss: 0.2584 - mae: 0.1881 - accuracy: 1.0000 - Recall: 0.9084 - Precision: 0.8803 - Specificity: 0.9985 - AUC: 0.9648
    Epoch 1: val_accuracy improved from -inf to 1.00000, saving model to best_model.h5
    245/245 [==============================] - 3s 6ms/step - loss: 0.2556 - mae: 0.1861 - accuracy: 1.0000 - Recall: 0.9094 - Precision: 0.8820 - Specificity: 0.9985 - AUC: 0.9656 - val_loss: 0.1321 - val_mae: 0.0886 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.8854 - val_Specificity: 1.0000 - val_AUC: 0.9983
    Epoch 2/20
     46/245 [====>.........................] - ETA: 0s - loss: 0.1000 - mae: 0.0732 - accuracy: 1.0000 - Recall: 0.9593 - Precision: 0.9579 - Specificity: 1.0000 - AUC: 0.9951

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    

    244/245 [============================>.] - ETA: 0s - loss: 0.0763 - mae: 0.0565 - accuracy: 1.0000 - Recall: 0.9696 - Precision: 0.9714 - Specificity: 1.0000 - AUC: 0.9973
    Epoch 2: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0763 - mae: 0.0565 - accuracy: 1.0000 - Recall: 0.9695 - Precision: 0.9715 - Specificity: 1.0000 - AUC: 0.9973 - val_loss: 0.0474 - val_mae: 0.0382 - val_accuracy: 1.0000 - val_Recall: 0.9916 - val_Precision: 0.9804 - val_Specificity: 1.0000 - val_AUC: 0.9995
    Epoch 3/20
    245/245 [==============================] - ETA: 0s - loss: 0.0616 - mae: 0.0420 - accuracy: 1.0000 - Recall: 0.9715 - Precision: 0.9746 - Specificity: 1.0000 - AUC: 0.9978
    Epoch 3: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0616 - mae: 0.0420 - accuracy: 1.0000 - Recall: 0.9715 - Precision: 0.9746 - Specificity: 1.0000 - AUC: 0.9978 - val_loss: 0.0379 - val_mae: 0.0303 - val_accuracy: 1.0000 - val_Recall: 0.9802 - val_Precision: 0.9968 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 4/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0449 - mae: 0.0315 - accuracy: 1.0000 - Recall: 0.9825 - Precision: 0.9820 - Specificity: 1.0000 - AUC: 0.9989
    Epoch 4: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0451 - mae: 0.0315 - accuracy: 1.0000 - Recall: 0.9820 - Precision: 0.9822 - Specificity: 1.0000 - AUC: 0.9989 - val_loss: 0.0290 - val_mae: 0.0235 - val_accuracy: 1.0000 - val_Recall: 0.9937 - val_Precision: 0.9896 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 5/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0389 - mae: 0.0270 - accuracy: 1.0000 - Recall: 0.9833 - Precision: 0.9851 - Specificity: 1.0000 - AUC: 0.9992
    Epoch 5: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 7ms/step - loss: 0.0386 - mae: 0.0269 - accuracy: 1.0000 - Recall: 0.9836 - Precision: 0.9854 - Specificity: 1.0000 - AUC: 0.9992 - val_loss: 0.0256 - val_mae: 0.0204 - val_accuracy: 1.0000 - val_Recall: 0.9958 - val_Precision: 0.9886 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 6/20
    237/245 [============================>.] - ETA: 0s - loss: 0.0308 - mae: 0.0219 - accuracy: 1.0000 - Recall: 0.9881 - Precision: 0.9863 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 6: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 6ms/step - loss: 0.0313 - mae: 0.0222 - accuracy: 1.0000 - Recall: 0.9877 - Precision: 0.9859 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0254 - val_mae: 0.0194 - val_accuracy: 1.0000 - val_Recall: 0.9791 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 7/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0381 - mae: 0.0229 - accuracy: 1.0000 - Recall: 0.9840 - Precision: 0.9830 - Specificity: 1.0000 - AUC: 0.9990
    Epoch 7: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0380 - mae: 0.0228 - accuracy: 1.0000 - Recall: 0.9838 - Precision: 0.9833 - Specificity: 1.0000 - AUC: 0.9990 - val_loss: 0.0201 - val_mae: 0.0162 - val_accuracy: 1.0000 - val_Recall: 0.9969 - val_Precision: 0.9907 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 8/20
    234/245 [===========================>..] - ETA: 0s - loss: 0.0300 - mae: 0.0200 - accuracy: 1.0000 - Recall: 0.9857 - Precision: 0.9855 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 8: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0297 - mae: 0.0199 - accuracy: 1.0000 - Recall: 0.9862 - Precision: 0.9857 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0175 - val_mae: 0.0142 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9907 - val_Specificity: 1.0000 - val_AUC: 1.0000
    Epoch 9/20
    235/245 [===========================>..] - ETA: 0s - loss: 0.0338 - mae: 0.0207 - accuracy: 1.0000 - Recall: 0.9845 - Precision: 0.9861 - Specificity: 0.9997 - AUC: 0.9992
    Epoch 9: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0339 - mae: 0.0208 - accuracy: 1.0000 - Recall: 0.9843 - Precision: 0.9864 - Specificity: 0.9998 - AUC: 0.9992 - val_loss: 0.0428 - val_mae: 0.0231 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9599 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 10/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0428 - mae: 0.0226 - accuracy: 1.0000 - Recall: 0.9816 - Precision: 0.9829 - Specificity: 1.0000 - AUC: 0.9989
    Epoch 10: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0427 - mae: 0.0227 - accuracy: 1.0000 - Recall: 0.9820 - Precision: 0.9825 - Specificity: 1.0000 - AUC: 0.9989 - val_loss: 0.0213 - val_mae: 0.0163 - val_accuracy: 1.0000 - val_Recall: 0.9896 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 11/20
    233/245 [===========================>..] - ETA: 0s - loss: 0.0323 - mae: 0.0196 - accuracy: 1.0000 - Recall: 0.9851 - Precision: 0.9868 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 11: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0319 - mae: 0.0194 - accuracy: 1.0000 - Recall: 0.9856 - Precision: 0.9869 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0231 - val_mae: 0.0163 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9806 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 12/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0241 - mae: 0.0166 - accuracy: 1.0000 - Recall: 0.9894 - Precision: 0.9909 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 12: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0242 - mae: 0.0167 - accuracy: 1.0000 - Recall: 0.9893 - Precision: 0.9906 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0148 - val_mae: 0.0120 - val_accuracy: 1.0000 - val_Recall: 0.9927 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 1.0000
    Epoch 13/20
    244/245 [============================>.] - ETA: 0s - loss: 0.0292 - mae: 0.0173 - accuracy: 1.0000 - Recall: 0.9877 - Precision: 0.9869 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 13: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0291 - mae: 0.0173 - accuracy: 1.0000 - Recall: 0.9877 - Precision: 0.9870 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0176 - val_mae: 0.0132 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9886 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 14/20
    236/245 [===========================>..] - ETA: 0s - loss: 0.0347 - mae: 0.0193 - accuracy: 1.0000 - Recall: 0.9843 - Precision: 0.9854 - Specificity: 0.9997 - AUC: 0.9992
    Epoch 14: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0341 - mae: 0.0191 - accuracy: 1.0000 - Recall: 0.9849 - Precision: 0.9859 - Specificity: 0.9998 - AUC: 0.9992 - val_loss: 0.0247 - val_mae: 0.0166 - val_accuracy: 1.0000 - val_Recall: 0.9791 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 15/20
    244/245 [============================>.] - ETA: 0s - loss: 0.0230 - mae: 0.0152 - accuracy: 1.0000 - Recall: 0.9887 - Precision: 0.9905 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 15: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0229 - mae: 0.0152 - accuracy: 1.0000 - Recall: 0.9888 - Precision: 0.9906 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0232 - val_mae: 0.0155 - val_accuracy: 1.0000 - val_Recall: 0.9791 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 16/20
    236/245 [===========================>..] - ETA: 0s - loss: 0.0258 - mae: 0.0155 - accuracy: 1.0000 - Recall: 0.9897 - Precision: 0.9886 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 16: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 6ms/step - loss: 0.0258 - mae: 0.0155 - accuracy: 1.0000 - Recall: 0.9890 - Precision: 0.9890 - Specificity: 1.0000 - AUC: 0.9996 - val_loss: 0.0177 - val_mae: 0.0127 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9846 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 17/20
    239/245 [============================>.] - ETA: 0s - loss: 0.0227 - mae: 0.0145 - accuracy: 1.0000 - Recall: 0.9893 - Precision: 0.9896 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 17: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 6ms/step - loss: 0.0224 - mae: 0.0144 - accuracy: 1.0000 - Recall: 0.9896 - Precision: 0.9896 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0134 - val_mae: 0.0106 - val_accuracy: 1.0000 - val_Recall: 0.9948 - val_Precision: 0.9979 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 18/20
    244/245 [============================>.] - ETA: 0s - loss: 0.0333 - mae: 0.0168 - accuracy: 1.0000 - Recall: 0.9874 - Precision: 0.9866 - Specificity: 1.0000 - AUC: 0.9991
    Epoch 18: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0333 - mae: 0.0169 - accuracy: 1.0000 - Recall: 0.9875 - Precision: 0.9867 - Specificity: 1.0000 - AUC: 0.9991 - val_loss: 0.0269 - val_mae: 0.0172 - val_accuracy: 1.0000 - val_Recall: 0.9708 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 19/20
    237/245 [============================>.] - ETA: 0s - loss: 0.0426 - mae: 0.0219 - accuracy: 1.0000 - Recall: 0.9820 - Precision: 0.9846 - Specificity: 0.9997 - AUC: 0.9985
    Epoch 19: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0420 - mae: 0.0218 - accuracy: 1.0000 - Recall: 0.9825 - Precision: 0.9846 - Specificity: 0.9998 - AUC: 0.9985 - val_loss: 0.0241 - val_mae: 0.0167 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9825 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 20/20
    234/245 [===========================>..] - ETA: 0s - loss: 0.0249 - mae: 0.0159 - accuracy: 1.0000 - Recall: 0.9885 - Precision: 0.9899 - Specificity: 1.0000 - AUC: 0.9996
    Epoch 20: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0248 - mae: 0.0159 - accuracy: 1.0000 - Recall: 0.9885 - Precision: 0.9898 - Specificity: 1.0000 - AUC: 0.9996 - val_loss: 0.0215 - val_mae: 0.0144 - val_accuracy: 1.0000 - val_Recall: 0.9823 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9999
    62/62 [==============================] - 0s 2ms/step
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_51_15.png)
    


    Accuracy: 0.9739663093415007
    Precision: 1.0
    Recall: 0.9467640918580376
    F1-score: 0.9726541554959786
    ROC AUC: 0.9733820459290188
    Epoch 1/20
    234/245 [===========================>..] - ETA: 0s - loss: 0.3023 - mae: 0.2215 - accuracy: 1.0000 - Recall: 0.8213 - Precision: 0.9004 - Specificity: 0.9958 - AUC: 0.9528
    Epoch 1: val_accuracy improved from -inf to 1.00000, saving model to best_model.h5
    245/245 [==============================] - 3s 8ms/step - loss: 0.2936 - mae: 0.2154 - accuracy: 1.0000 - Recall: 0.8280 - Precision: 0.9037 - Specificity: 0.9965 - AUC: 0.9557 - val_loss: 0.1306 - val_mae: 0.0933 - val_accuracy: 1.0000 - val_Recall: 0.9979 - val_Precision: 0.8959 - val_Specificity: 1.0000 - val_AUC: 0.9979
    Epoch 2/20
     28/245 [==>...........................] - ETA: 0s - loss: 0.1110 - mae: 0.0854 - accuracy: 1.0000 - Recall: 0.9569 - Precision: 0.9569 - Specificity: 1.0000 - AUC: 0.9956

    /usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3079: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
      saving_api.save_model(
    

    231/245 [===========================>..] - ETA: 0s - loss: 0.0874 - mae: 0.0658 - accuracy: 1.0000 - Recall: 0.9654 - Precision: 0.9689 - Specificity: 1.0000 - AUC: 0.9967
    Epoch 2: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0859 - mae: 0.0646 - accuracy: 1.0000 - Recall: 0.9663 - Precision: 0.9691 - Specificity: 1.0000 - AUC: 0.9968 - val_loss: 0.0561 - val_mae: 0.0432 - val_accuracy: 1.0000 - val_Recall: 0.9572 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 3/20
    241/245 [============================>.] - ETA: 0s - loss: 0.0593 - mae: 0.0436 - accuracy: 1.0000 - Recall: 0.9764 - Precision: 0.9758 - Specificity: 1.0000 - AUC: 0.9983
    Epoch 3: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0590 - mae: 0.0434 - accuracy: 1.0000 - Recall: 0.9768 - Precision: 0.9763 - Specificity: 1.0000 - AUC: 0.9983 - val_loss: 0.0393 - val_mae: 0.0313 - val_accuracy: 1.0000 - val_Recall: 0.9916 - val_Precision: 0.9865 - val_Specificity: 1.0000 - val_AUC: 0.9996
    Epoch 4/20
    231/245 [===========================>..] - ETA: 0s - loss: 0.0506 - mae: 0.0355 - accuracy: 1.0000 - Recall: 0.9786 - Precision: 0.9808 - Specificity: 1.0000 - AUC: 0.9986
    Epoch 4: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0520 - mae: 0.0359 - accuracy: 1.0000 - Recall: 0.9786 - Precision: 0.9799 - Specificity: 1.0000 - AUC: 0.9985 - val_loss: 0.0346 - val_mae: 0.0262 - val_accuracy: 1.0000 - val_Recall: 0.9718 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 5/20
    244/245 [============================>.] - ETA: 0s - loss: 0.0361 - mae: 0.0266 - accuracy: 1.0000 - Recall: 0.9856 - Precision: 0.9856 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 5: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0365 - mae: 0.0268 - accuracy: 1.0000 - Recall: 0.9856 - Precision: 0.9851 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0513 - val_mae: 0.0321 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9541 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 6/20
    242/245 [============================>.] - ETA: 0s - loss: 0.0377 - mae: 0.0260 - accuracy: 1.0000 - Recall: 0.9844 - Precision: 0.9857 - Specificity: 1.0000 - AUC: 0.9991
    Epoch 6: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 6ms/step - loss: 0.0376 - mae: 0.0259 - accuracy: 1.0000 - Recall: 0.9843 - Precision: 0.9859 - Specificity: 1.0000 - AUC: 0.9991 - val_loss: 0.0284 - val_mae: 0.0199 - val_accuracy: 1.0000 - val_Recall: 0.9969 - val_Precision: 0.9775 - val_Specificity: 1.0000 - val_AUC: 0.9997
    Epoch 7/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0304 - mae: 0.0213 - accuracy: 1.0000 - Recall: 0.9888 - Precision: 0.9857 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 7: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 6ms/step - loss: 0.0304 - mae: 0.0213 - accuracy: 1.0000 - Recall: 0.9883 - Precision: 0.9859 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0271 - val_mae: 0.0194 - val_accuracy: 1.0000 - val_Recall: 0.9979 - val_Precision: 0.9765 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 8/20
    244/245 [============================>.] - ETA: 0s - loss: 0.0353 - mae: 0.0219 - accuracy: 1.0000 - Recall: 0.9848 - Precision: 0.9840 - Specificity: 1.0000 - AUC: 0.9992
    Epoch 8: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0353 - mae: 0.0219 - accuracy: 1.0000 - Recall: 0.9846 - Precision: 0.9841 - Specificity: 1.0000 - AUC: 0.9992 - val_loss: 0.0210 - val_mae: 0.0161 - val_accuracy: 1.0000 - val_Recall: 0.9864 - val_Precision: 0.9989 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 9/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0289 - mae: 0.0195 - accuracy: 1.0000 - Recall: 0.9871 - Precision: 0.9876 - Specificity: 1.0000 - AUC: 0.9995
    Epoch 9: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0287 - mae: 0.0194 - accuracy: 1.0000 - Recall: 0.9872 - Precision: 0.9877 - Specificity: 1.0000 - AUC: 0.9995 - val_loss: 0.0193 - val_mae: 0.0144 - val_accuracy: 1.0000 - val_Recall: 0.9948 - val_Precision: 0.9927 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 10/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0311 - mae: 0.0198 - accuracy: 1.0000 - Recall: 0.9871 - Precision: 0.9858 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 10: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0309 - mae: 0.0197 - accuracy: 1.0000 - Recall: 0.9872 - Precision: 0.9859 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0213 - val_mae: 0.0150 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9846 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 11/20
    243/245 [============================>.] - ETA: 0s - loss: 0.0321 - mae: 0.0192 - accuracy: 1.0000 - Recall: 0.9869 - Precision: 0.9850 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 11: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 5ms/step - loss: 0.0321 - mae: 0.0193 - accuracy: 1.0000 - Recall: 0.9867 - Precision: 0.9851 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0191 - val_mae: 0.0141 - val_accuracy: 1.0000 - val_Recall: 0.9885 - val_Precision: 0.9947 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 12/20
    236/245 [===========================>..] - ETA: 0s - loss: 0.0387 - mae: 0.0213 - accuracy: 1.0000 - Recall: 0.9808 - Precision: 0.9840 - Specificity: 1.0000 - AUC: 0.9991
    Epoch 12: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0386 - mae: 0.0214 - accuracy: 1.0000 - Recall: 0.9815 - Precision: 0.9833 - Specificity: 1.0000 - AUC: 0.9991 - val_loss: 0.0358 - val_mae: 0.0207 - val_accuracy: 1.0000 - val_Recall: 0.9592 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 13/20
    235/245 [===========================>..] - ETA: 0s - loss: 0.0228 - mae: 0.0156 - accuracy: 1.0000 - Recall: 0.9905 - Precision: 0.9913 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 13: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0235 - mae: 0.0162 - accuracy: 1.0000 - Recall: 0.9903 - Precision: 0.9906 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0722 - val_mae: 0.0298 - val_accuracy: 1.0000 - val_Recall: 0.9436 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 14/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0294 - mae: 0.0175 - accuracy: 1.0000 - Recall: 0.9870 - Precision: 0.9885 - Specificity: 1.0000 - AUC: 0.9994
    Epoch 14: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0293 - mae: 0.0176 - accuracy: 1.0000 - Recall: 0.9872 - Precision: 0.9885 - Specificity: 1.0000 - AUC: 0.9994 - val_loss: 0.0161 - val_mae: 0.0122 - val_accuracy: 1.0000 - val_Recall: 1.0000 - val_Precision: 0.9927 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 15/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0220 - mae: 0.0149 - accuracy: 1.0000 - Recall: 0.9888 - Precision: 0.9893 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 15: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0225 - mae: 0.0151 - accuracy: 1.0000 - Recall: 0.9880 - Precision: 0.9895 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0196 - val_mae: 0.0125 - val_accuracy: 1.0000 - val_Recall: 0.9990 - val_Precision: 0.9835 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 16/20
    239/245 [============================>.] - ETA: 0s - loss: 0.0217 - mae: 0.0144 - accuracy: 1.0000 - Recall: 0.9901 - Precision: 0.9896 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 16: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0222 - mae: 0.0147 - accuracy: 1.0000 - Recall: 0.9901 - Precision: 0.9888 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0408 - val_mae: 0.0198 - val_accuracy: 1.0000 - val_Recall: 0.9624 - val_Precision: 1.0000 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 17/20
    238/245 [============================>.] - ETA: 0s - loss: 0.0285 - mae: 0.0158 - accuracy: 1.0000 - Recall: 0.9879 - Precision: 0.9895 - Specificity: 0.9997 - AUC: 0.9993
    Epoch 17: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 1s 4ms/step - loss: 0.0283 - mae: 0.0157 - accuracy: 1.0000 - Recall: 0.9883 - Precision: 0.9893 - Specificity: 0.9998 - AUC: 0.9993 - val_loss: 0.0205 - val_mae: 0.0135 - val_accuracy: 1.0000 - val_Recall: 0.9979 - val_Precision: 0.9856 - val_Specificity: 1.0000 - val_AUC: 0.9998
    Epoch 18/20
    238/245 [============================>.] - ETA: 0s - loss: 0.0200 - mae: 0.0133 - accuracy: 1.0000 - Recall: 0.9906 - Precision: 0.9914 - Specificity: 0.9997 - AUC: 0.9996
    Epoch 18: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 7ms/step - loss: 0.0199 - mae: 0.0133 - accuracy: 1.0000 - Recall: 0.9906 - Precision: 0.9916 - Specificity: 0.9998 - AUC: 0.9997 - val_loss: 0.0170 - val_mae: 0.0118 - val_accuracy: 1.0000 - val_Recall: 0.9885 - val_Precision: 0.9979 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 19/20
    242/245 [============================>.] - ETA: 0s - loss: 0.0229 - mae: 0.0140 - accuracy: 1.0000 - Recall: 0.9894 - Precision: 0.9897 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 19: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 8ms/step - loss: 0.0229 - mae: 0.0140 - accuracy: 1.0000 - Recall: 0.9893 - Precision: 0.9898 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0151 - val_mae: 0.0107 - val_accuracy: 1.0000 - val_Recall: 0.9927 - val_Precision: 0.9979 - val_Specificity: 1.0000 - val_AUC: 0.9999
    Epoch 20/20
    240/245 [============================>.] - ETA: 0s - loss: 0.0215 - mae: 0.0128 - accuracy: 1.0000 - Recall: 0.9909 - Precision: 0.9912 - Specificity: 1.0000 - AUC: 0.9997
    Epoch 20: val_accuracy did not improve from 1.00000
    245/245 [==============================] - 2s 7ms/step - loss: 0.0214 - mae: 0.0128 - accuracy: 1.0000 - Recall: 0.9911 - Precision: 0.9909 - Specificity: 1.0000 - AUC: 0.9997 - val_loss: 0.0144 - val_mae: 0.0102 - val_accuracy: 1.0000 - val_Recall: 0.9979 - val_Precision: 0.9927 - val_Specificity: 1.0000 - val_AUC: 0.9999
    62/62 [==============================] - 0s 2ms/step
    


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_51_19.png)
    


    Accuracy: 0.9897907095456866
    Precision: 1.0
    Recall: 0.9791013584117032
    F1-score: 0.9894403379091868
    ROC AUC: 0.9895506792058516
    


```python
best_model = keras.models.load_model(name_checkpoint_best_model)
```


```python
X_test_normalized = scaler.transform(X_test)
y_pred = best_model.predict(X_test_normalized)

y_pred_binary = (y_pred > 0.9).astype(int)
```

    35/35 [==============================] - 0s 2ms/step
    


```python
print('Métricas gerais (média) para a predição dos 5 folders com Rede Neural:\n')
calcule_metrics_classification(y_test, y_pred_binary)
```

    Métricas gerais (média) para a predição dos 5 folders com Rede Neural:
    
    Accuracy: 0.9412304866850322
    Precision: 1.0
    Recall: 0.8783269961977186
    F1-score: 0.9352226720647774
    ROC AUC: 0.9391634980988592
    


```python
generate_matriz_confusion(y_test, y_pred_binary)
```


    
![png](Trabalho_Machine_Learning_files/Trabalho_Machine_Learning_55_0.png)
    



```python

```
