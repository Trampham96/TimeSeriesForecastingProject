# **Task 1: Combine CSV Files into a Single DataFrame**

# 1. Read the data


```python
# prompt: connect to google drive
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
%cd /content/drive/MyDrive/6000_01/Week12(19112924)
```

    /content/drive/MyDrive/6000_01/Week12(19112924)



```python
%cd Datasets/
```

    /content/drive/MyDrive/6000_01/Week12(19112924)/Datasets



```python
ls
```

    combined_dataframe.csv  [0m[01;34mdata[0m/  data.zip



```python
!unzip data.zip -d 'data'
```

    Archive:  data.zip
       creating: data/data/
      inflating: data/data/Oranges (big size).csv  
      inflating: data/data/Sorghum.csv   
      inflating: data/data/Peas (fresh).csv  
      inflating: data/data/Beans (dry).csv  
      inflating: data/data/Cassava.csv   
      inflating: data/data/Maize.csv     
      inflating: data/data/Potatoes (Irish).csv  
      inflating: data/data/Tomatoes.csv  
      inflating: data/data/Chili (red).csv  


# 2. Load all the CSV files into separate DataFrames & Extract the name of the good from the file name


```python
import os
import pandas as pd
```


```python
# Path to the directory containing CSV files
csv_folder = "data/data"

# List all CSV files in the directory
csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

# Print the list of CSV files
print("CSV files found:", csv_files)
```

    CSV files found: ['Oranges (big size).csv', 'Sorghum.csv', 'Peas (fresh).csv', 'Beans (dry).csv', 'Cassava.csv', 'Maize.csv', 'Potatoes (Irish).csv', 'Tomatoes.csv', 'Chili (red).csv']



```python
# Dictionary to store DataFrames
dataframes = {}

# Loop through each CSV file
for file in csv_files:
    # Extract the name of the good (remove '.csv' from file name)
    good_name = os.path.splitext(file)[0]

    # Construct the full file path
    file_path = os.path.join(csv_folder, file)

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    print(f"Loaded data for: {good_name}")  # Confirmation message

    # Store the DataFrame in the dictionary
    dataframes[good_name] = df
```

    Loaded data for: Oranges (big size)
    Loaded data for: Sorghum
    Loaded data for: Peas (fresh)
    Loaded data for: Beans (dry)
    Loaded data for: Cassava
    Loaded data for: Maize
    Loaded data for: Potatoes (Irish)
    Loaded data for: Tomatoes
    Loaded data for: Chili (red)



```python
print("Loaded DataFrames:", dataframes.keys())
```

    Loaded DataFrames: dict_keys(['Oranges (big size)', 'Sorghum', 'Peas (fresh)', 'Beans (dry)', 'Cassava', 'Maize', 'Potatoes (Irish)', 'Tomatoes', 'Chili (red)'])


# 3. Ensure the date column is parsed correctly as a datetime object


```python
for good_name, df in dataframes.items():
    # Ensure the date column is parsed correctly as a datetime object
    df['date'] = pd.to_datetime(df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str) + '-01')

    # Keep only the relevant columns: 'date' and 'mp_price'
    df = df[['date', 'mp_price']].rename(columns={'mp_price': good_name})

    # Set 'date' as the index
    df.set_index('date', inplace=True)

    # Update the DataFrame in the dictionary
    dataframes[good_name] = df
```


```python
for good_name, df in dataframes.items():
    # Group by 'date' and calculate the sum
    df = df.groupby('date').sum()
    dataframes[good_name] = df
```

#4. Combine all DataFrames into a single DataFrame & Align dates across all goods (fill any missing values with NaN)


```python
# Combine all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes.values(), axis=1)

# Align dates across all goods (fill missing values with NaN)
combined_df.sort_index(inplace=True)
```

#5. Export and Display the Data


```python
from IPython.display import FileLink

# Provide the file path
output_file = "combined_dataframe.csv"
combined_df.to_csv(output_file)

# Create a download link
print(f"Download your file: {output_file}")
FileLink(output_file)
```

    Download your file: combined_dataframe.csv





<a href='combined_dataframe.csv' target='_blank'>combined_dataframe.csv</a><br>




```python
# Display the first 05 rows of the combined DataFrame
print(combined_df.head())
```

                Oranges (big size)  Sorghum  Peas (fresh)  Beans (dry)  Cassava  \
    date                                                                          
    2008-01-01          20003.5024  5713.75    20493.7778      8987.50  3731.25   
    2008-02-01          20003.5024  5277.50    20493.7778      8482.50  3490.00   
    2008-03-01          20003.5024  5446.25    20493.7778      8992.50  3550.00   
    2008-04-01          20003.5024  5656.25    20493.7778      9311.25  4055.00   
    2008-05-01          20003.5024  6303.00    20493.7778      9372.50  3657.50   
    
                  Maize  Potatoes (Irish)    Tomatoes  Chili (red)  
    date                                                            
    2008-01-01  5813.75           3201.00  15399.6762   46284.9433  
    2008-02-01  5038.75           3251.25  15399.6762   46284.9433  
    2008-03-01  4666.25           3228.75  15399.6762   46284.9433  
    2008-04-01  5148.75           3336.25  15399.6762   46284.9433  
    2008-05-01  4899.25           3254.25  15399.6762   46284.9433  


#**Task 2: Explore the Consolidated Data**

# 1. Data Overview


```python
# Display the shape of the DataFrame
print("Shape of the DataFrame (rows, columns):", combined_df.shape)

# List column names and their data types
print("\nColumn Names and Data Types:")
print(combined_df.dtypes)

# Verify the date range of the index
print("\nDate Range of the Index:")
print(f"Start: {combined_df.index.min()}, End: {combined_df.index.max()}")

# Check if the date index is continuous
print("\nIs the date index continuous?")
date_diff = combined_df.index.to_series().diff().dropna()  # Compute differences between dates
print(date_diff.unique())  # Show unique differences in the index
```

    Shape of the DataFrame (rows, columns): (96, 9)
    
    Column Names and Data Types:
    Oranges (big size)    float64
    Sorghum               float64
    Peas (fresh)          float64
    Beans (dry)           float64
    Cassava               float64
    Maize                 float64
    Potatoes (Irish)      float64
    Tomatoes              float64
    Chili (red)           float64
    dtype: object
    
    Date Range of the Index:
    Start: 2008-01-01 00:00:00, End: 2015-12-01 00:00:00
    
    Is the date index continuous?
    <TimedeltaArray>
    ['31 days', '29 days', '30 days', '28 days']
    Length: 4, dtype: timedelta64[ns]


# 2. Missing Values


```python
# Count missing values for each column
missing_values = combined_df.isnull().sum()

# Percentage of missing values per column
missing_percentage = (missing_values / len(combined_df)) * 100

# Display missing values summary
print("\nMissing Values Summary:")
missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Percentage (%)": missing_percentage
})
print(missing_summary)
```

    
    Missing Values Summary:
                        Missing Values  Percentage (%)
    Oranges (big size)              58       60.416667
    Sorghum                          0        0.000000
    Peas (fresh)                    36       37.500000
    Beans (dry)                      0        0.000000
    Cassava                          0        0.000000
    Maize                            0        0.000000
    Potatoes (Irish)                 0        0.000000
    Tomatoes                        58       60.416667
    Chili (red)                     58       60.416667


# 3. Descriptive Statistics


```python
# Generate summary statistics
print("\nDescriptive Statistics:")
print(combined_df.describe().T)  # Transpose for better readability
```

    
    Descriptive Statistics:
                        count          mean           std         min  \
    Oranges (big size)   38.0  28065.174542   4318.177902  18818.5608   
    Sorghum              96.0  12093.232120   4784.504492   5277.5000   
    Peas (fresh)         60.0  27833.385982   5977.237271  16805.3692   
    Beans (dry)          96.0  15670.003506   6481.071688   6789.1474   
    Cassava              96.0   5759.139928   1843.383171   3093.7500   
    Maize                96.0   9714.711537   3417.871514   4302.5416   
    Potatoes (Irish)     96.0   7316.631144   3140.831956   3201.0000   
    Tomatoes             38.0  21211.048087   3402.436358  11443.7596   
    Chili (red)          38.0  57090.394029  15143.250930  34805.1570   
    
                                 25%          50%           75%          max  
    Oranges (big size)  25660.827300  29110.37595  30450.768350   34834.4308  
    Sorghum              7882.677075  11593.35350  16908.399050   20995.7226  
    Peas (fresh)        23173.175500  28381.53855  31468.167475   43572.1722  
    Beans (dry)         10254.239575  13584.58345  21289.660500   32332.6222  
    Cassava              4037.923675   5363.65180   7605.112575    9130.3723  
    Maize                6296.531250   9425.79780  12713.183850   17639.8346  
    Potatoes (Irish)     4114.562500   6955.50890  10101.092050   14482.6370  
    Tomatoes            18789.377400  21162.99210  23354.322550   27121.4215  
    Chili (red)         47587.871900  56124.98930  61339.981650  104779.4088  


# 4. Time Series Visualizations


```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define a color palette (one color per good)
goods = combined_df.columns
colors = cm.tab10.colors[:len(goods)]  # Use matplotlib's tab10 color map
color_mapping = dict(zip(goods, colors))  # Map each good to a specific color
# Plot each good's time series with its assigned color
for column in goods:
    plt.figure(figsize=(10, 5))
    plt.plot(combined_df.index, combined_df[column], label=column, color=color_mapping[column])
    plt.title(f"Time Series for {column}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()
```


    
![png](output_28_0.png)
    



    
![png](output_28_1.png)
    



    
![png](output_28_2.png)
    



    
![png](output_28_3.png)
    



    
![png](output_28_4.png)
    



    
![png](output_28_5.png)
    



    
![png](output_28_6.png)
    



    
![png](output_28_7.png)
    



    
![png](output_28_8.png)
    



```python
# Overlay all time series in a single plot
plt.figure(figsize=(15, 7))
for column in goods:
    plt.plot(combined_df.index, combined_df[column], label=column, color=color_mapping[column])

plt.title("Overlay of Time Series for All Goods")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()
```


    
![png](output_29_0.png)
    


# **Task 3: Handle Missing Values**

# 1. Identify Missing Values


```python
# Count missing values for each column
missing_values = combined_df.isnull().sum()

# Total number of missing entries
total_missing = missing_values.sum()

# Print the missing values summary
print("Missing Values Summary:")
print(missing_values)

# Total missing entries
print(f"\nTotal missing entries: {total_missing}")
```

    Missing Values Summary:
    Oranges (big size)    58
    Sorghum                0
    Peas (fresh)          36
    Beans (dry)            0
    Cassava                0
    Maize                  0
    Potatoes (Irish)       0
    Tomatoes              58
    Chili (red)           58
    dtype: int64
    
    Total missing entries: 210


# 2. Handle Missing Data


```python
# Apply linear interpolation to handle missing values
combined_df = combined_df.interpolate(method='linear')

# Verify no missing values remain
print("\nMissing Values After Interpolation:")
print(combined_df.isnull().sum())
```

    
    Missing Values After Interpolation:
    Oranges (big size)    48
    Sorghum                0
    Peas (fresh)          36
    Beans (dry)            0
    Cassava                0
    Maize                  0
    Potatoes (Irish)       0
    Tomatoes              48
    Chili (red)           48
    dtype: int64



```python
# Apply forward-fill, then backward-fill for edge missing values
combined_df.ffill(inplace=True)
combined_df.bfill(inplace=True)
# Verify no missing values remain
print("\nMissing Values After Filling Edges:")
print(combined_df.isnull().sum())

```

    
    Missing Values After Filling Edges:
    Oranges (big size)    0
    Sorghum               0
    Peas (fresh)          0
    Beans (dry)           0
    Cassava               0
    Maize                 0
    Potatoes (Irish)      0
    Tomatoes              0
    Chili (red)           0
    dtype: int64



```python
# Display the first 05 rows of the combined DataFrame
print(combined_df.head())
```

                Oranges (big size)  Sorghum  Peas (fresh)  Beans (dry)  Cassava  \
    date                                                                          
    2008-01-01          20003.5024  5713.75    20493.7778      8987.50  3731.25   
    2008-02-01          20003.5024  5277.50    20493.7778      8482.50  3490.00   
    2008-03-01          20003.5024  5446.25    20493.7778      8992.50  3550.00   
    2008-04-01          20003.5024  5656.25    20493.7778      9311.25  4055.00   
    2008-05-01          20003.5024  6303.00    20493.7778      9372.50  3657.50   
    
                  Maize  Potatoes (Irish)    Tomatoes  Chili (red)  
    date                                                            
    2008-01-01  5813.75           3201.00  15399.6762   46284.9433  
    2008-02-01  5038.75           3251.25  15399.6762   46284.9433  
    2008-03-01  4666.25           3228.75  15399.6762   46284.9433  
    2008-04-01  5148.75           3336.25  15399.6762   46284.9433  
    2008-05-01  4899.25           3254.25  15399.6762   46284.9433  


# 3. Justification

**I first used linear interpolation to estimate missing values by utilizing surrounding data points, preserving trends, and ensuring smooth changes, which is ideal for time-series data. However, I observed that interpolation could not handle missing values at the start or end of the series due to the lack of preceding or succeeding values. To address this, I applied forward-fill (ffill) to propagate the last known value forward for missing data at the start and backward-fill (bfill) to fill missing values at the end with the nearest future value. This hybrid approach allowed me to ensure a complete dataset while maintaining trends and continuity.**

# **Task 4: Analyze Similarities Between Products**

# 1. Calculate the Correlation Matrix


```python
# Calculate the correlation matrix
correlation_matrix = combined_df.corr()

# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
```

    Correlation Matrix:
                        Oranges (big size)   Sorghum  Peas (fresh)  Beans (dry)  \
    Oranges (big size)            1.000000  0.738297      0.775045     0.904047   
    Sorghum                       0.738297  1.000000      0.765992     0.793836   
    Peas (fresh)                  0.775045  0.765992      1.000000     0.817035   
    Beans (dry)                   0.904047  0.793836      0.817035     1.000000   
    Cassava                       0.803408  0.858476      0.763748     0.871119   
    Maize                         0.787192  0.836813      0.759185     0.894670   
    Potatoes (Irish)              0.857795  0.836812      0.746718     0.924052   
    Tomatoes                      0.700465  0.827614      0.669129     0.690752   
    Chili (red)                   0.542966  0.238046      0.285427     0.544317   
    
                         Cassava     Maize  Potatoes (Irish)  Tomatoes  \
    Oranges (big size)  0.803408  0.787192          0.857795  0.700465   
    Sorghum             0.858476  0.836813          0.836812  0.827614   
    Peas (fresh)        0.763748  0.759185          0.746718  0.669129   
    Beans (dry)         0.871119  0.894670          0.924052  0.690752   
    Cassava             1.000000  0.890414          0.891603  0.810000   
    Maize               0.890414  1.000000          0.850866  0.676072   
    Potatoes (Irish)    0.891603  0.850866          1.000000  0.796839   
    Tomatoes            0.810000  0.676072          0.796839  1.000000   
    Chili (red)         0.366543  0.324408          0.553148  0.237128   
    
                        Chili (red)  
    Oranges (big size)     0.542966  
    Sorghum                0.238046  
    Peas (fresh)           0.285427  
    Beans (dry)            0.544317  
    Cassava                0.366543  
    Maize                  0.324408  
    Potatoes (Irish)       0.553148  
    Tomatoes               0.237128  
    Chili (red)            1.000000  


# 2. Identify the Most Similar Pair of Goods


```python
# Unstack the correlation matrix to get pairs of goods
correlation_pairs = correlation_matrix.unstack()

# Sort by correlation values
sorted_correlation = correlation_pairs.sort_values(ascending=False)

# Remove self-correlation (correlation of goods with themselves)
most_similar_pair = sorted_correlation[sorted_correlation < 1].idxmax()

# Get the highest correlation value
highest_correlation = sorted_correlation[sorted_correlation < 1].max()

print(f"\nMost Similar Pair of Goods: {most_similar_pair}")
print(f"Highest Correlation Value: {highest_correlation}")
```

    
    Most Similar Pair of Goods: ('Potatoes (Irish)', 'Beans (dry)')
    Highest Correlation Value: 0.9240522923849255


# 3. Visualize the Correlation Matrix


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()
```


    
![png](output_45_0.png)
    


# 4. Justify Why Correlation is a Meaningful Metric

**The most similar pair of goods, Potatoes (Irish) and Beans (dry), exhibit a strong positive correlation of 0.924, indicating highly synchronized production patterns over time. Correlation is a meaningful metric for this comparison because it measures the strength and direction of the linear relationship between two variables, making it ideal for identifying similar trends in production. A high correlation suggests that these goods may be influenced by shared agricultural conditions (e.g., soil, weather), overlapping harvesting periods, or common external factors such as market demand or government policies. This insight can inform supply chain optimization, synchronized planning, and improved forecasting strategies for these goods.**

# **Task 5: Forecasting for the Next 6 Months**

# 1. Select a Good


```python
# Select the good for forecasting (e.g., Tomatoes)
selected_good = "Tomatoes"
selected_series = combined_df[selected_good]

# Display the last few values for context
print(f"Selected Good: {selected_good}")
print(selected_series.tail())
```

    Selected Good: Tomatoes
    date
    2015-08-01    22749.5612
    2015-09-01    26672.3094
    2015-10-01    22556.5164
    2015-11-01    21668.0348
    2015-12-01    20593.2964
    Freq: MS, Name: Tomatoes, dtype: float64


# 2. Moving Average Forecasting


```python
import matplotlib.pyplot as plt
import pandas as pd

# Moving Average Forecasting
window_sizes = [3, 6]  # Define window sizes (3 months, 6 months)
for window in window_sizes:
    # Calculate moving average
    moving_avg = selected_series.rolling(window=window).mean()

    # Plot the moving average
    plt.figure(figsize=(10, 5))
    plt.plot(selected_series, label="Actual")
    plt.plot(moving_avg, label=f"Moving Average (window={window})")
    plt.title(f"Moving Average Forecasting for {selected_good}")
    plt.xlabel("Date")
    plt.ylabel("Production Value")
    plt.legend()
    plt.grid()
    plt.show()

# Create a date range for the next 6 months
forecast_dates = pd.date_range(start=selected_series.index[-1] + pd.offsets.MonthBegin(1), periods=6, freq="MS")

# Predict the next 6 months by extending the moving average
last_window_avg = selected_series[-window_sizes[-1]:].mean()  # Average of the last window
moving_avg_forecast = pd.Series([last_window_avg] * 6, index=forecast_dates)

# Plot the extended forecast
plt.figure(figsize=(10, 5))
plt.plot(selected_series, label="Actual")
plt.plot(moving_avg_forecast, label="Moving Average Forecast", linestyle="--", color="red")
plt.title(f"Moving Average Forecast for {selected_good}")
plt.xlabel("Date")
plt.ylabel("Production Value")
plt.legend()
plt.grid()
plt.show()

# Print the forecasted values
print("Moving Average Forecast for the next 6 months:")
print(moving_avg_forecast)
```


    
![png](output_52_0.png)
    



    
![png](output_52_1.png)
    



    
![png](output_52_2.png)
    


    Moving Average Forecast for the next 6 months:
    2016-01-01    22473.007533
    2016-02-01    22473.007533
    2016-03-01    22473.007533
    2016-04-01    22473.007533
    2016-05-01    22473.007533
    2016-06-01    22473.007533
    Freq: MS, dtype: float64


# 3. Exponential Smoothing Forecasting


```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import pandas as pd

# Exponential Smoothing
model = ExponentialSmoothing(
    selected_series,
    trend="add",
    seasonal="add",
    seasonal_periods=12
)
fitted_model = model.fit()

# Forecast the next 6 months
exp_forecast = fitted_model.forecast(steps=6)

# Create a date range for the forecasted period
forecast_dates = pd.date_range(
    start=selected_series.index[-1] + pd.offsets.MonthBegin(1),  # Start from the next month
    periods=6,
    freq="MS"
)
exp_forecast = pd.Series(exp_forecast, index=forecast_dates)

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.plot(selected_series, label="Actual")
plt.plot(fitted_model.fittedvalues, label="Fitted")
plt.plot(exp_forecast, label="Forecast", linestyle="--", color="red")
plt.title(f"Exponential Smoothing Forecasting for {selected_good}")
plt.xlabel("Date")
plt.ylabel("Production Value")
plt.legend()
plt.grid()
plt.show()

# Print the forecasted values
print("Exponential Smoothing Forecast for the next 6 months:")
print(exp_forecast)
```

    /usr/local/lib/python3.10/dist-packages/statsmodels/tsa/holtwinters/model.py:918: ConvergenceWarning: Optimization failed to converge. Check mle_retvals.
      warnings.warn(



    
![png](output_54_1.png)
    


    Exponential Smoothing Forecast for the next 6 months:
    2016-01-01    21089.158705
    2016-02-01    21770.704358
    2016-03-01    22189.606341
    2016-04-01    22096.994279
    2016-05-01    22659.917718
    2016-06-01    21810.041997
    Freq: MS, dtype: float64


# 4. Facebook Prophet Forecasting


```python
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Prepare the data for Prophet
prophet_df = pd.DataFrame({"ds": selected_series.index, "y": selected_series.values})

# Initialize and fit the Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# Create a future dataframe for the next 6 months
future = prophet_model.make_future_dataframe(periods=6, freq="MS")  # Start from the next month

# Forecast the next 6 months
prophet_forecast = prophet_model.predict(future)

# Plot the forecast
prophet_model.plot(prophet_forecast)
plt.title(f"Prophet Forecasting for {selected_good}")
plt.xlabel("Date")
plt.ylabel("Production Value")
plt.show()

# Display the forecasted values
print("Prophet Forecast for the next 6 months:")
print(prophet_forecast[['ds', 'yhat']].tail(6))

```

    INFO:prophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:prophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    DEBUG:cmdstanpy:input tempfile: /tmp/tmpq5jdhizu/br5meldo.json
    DEBUG:cmdstanpy:input tempfile: /tmp/tmpq5jdhizu/swdk59a7.json
    DEBUG:cmdstanpy:idx 0
    DEBUG:cmdstanpy:running CmdStan, num_threads: None
    DEBUG:cmdstanpy:CmdStan args: ['/usr/local/lib/python3.10/dist-packages/prophet/stan_model/prophet_model.bin', 'random', 'seed=71663', 'data', 'file=/tmp/tmpq5jdhizu/br5meldo.json', 'init=/tmp/tmpq5jdhizu/swdk59a7.json', 'output', 'file=/tmp/tmpq5jdhizu/prophet_modelhrbh3jsv/prophet_model-20241125053244.csv', 'method=optimize', 'algorithm=newton', 'iter=10000']
    05:32:44 - cmdstanpy - INFO - Chain [1] start processing
    INFO:cmdstanpy:Chain [1] start processing
    05:32:45 - cmdstanpy - INFO - Chain [1] done processing
    INFO:cmdstanpy:Chain [1] done processing



    
![png](output_56_1.png)
    


    Prophet Forecast for the next 6 months:
                ds          yhat
    96  2016-01-01  22556.754845
    97  2016-02-01  24414.282929
    98  2016-03-01  24837.006145
    99  2016-04-01  24534.234095
    100 2016-05-01  25209.739440
    101 2016-06-01  24806.528385


# 5. Visualize All Forecasts Together


```python
# Combine the actual and forecasted data for comparison
plt.figure(figsize=(15, 7))
plt.plot(selected_series, label="Actual")

# Add forecasts
plt.plot(exp_forecast.index, exp_forecast, label="Exponential Smoothing")
plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label="Prophet Forecast")

# Title and Labels
plt.title(f"Comparison of Forecasting Methods for {selected_good}")
plt.xlabel("Date")
plt.ylabel("Production Value")
plt.legend()
plt.grid()
plt.show()
```


    
![png](output_58_0.png)
    

