# importing libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# importing visualization data libraries
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

sns.set(style = 'dark')

# title page
st.set_page_config(page_title="Air Quality Dashboard for Dicoding Course")

# load dataset
df = pd.read_csv('dashboard/main_data.csv')

# title of the dashboard
st.title('Air Quality Analysis at Shunyi Station :station:')

st.subheader('What is This Dashboard?') 
'Air Quality Analysis Dashboard presents analysis of air quality at Shunyi Station based on several parameters, such as PM2.5 and PM10 levels, Carbon Monoxide, Sulfur Trioxide, and many more!'

min_date = df['year'].min()
max_date = df['year'].max()

# sidebar configuration
st.sidebar.header('Please Select Year and Month')
input_year = st.sidebar.radio('Choose Year', list(sorted(df['year'].unique())))
input_month = st.sidebar.selectbox('Choose Month', list(sorted(df['month'].unique())))  

# filter data to shown changes in dashboard
df_filter = df[(df['year'] == input_year) & (df['month'] == input_month)].copy()

# descriptive statistic
st.subheader('Data Summary')
st.write(df_filter.describe())

def sum_sulfur(df):
    sulfur_df = df.pivot_table(values = 'SO2', index='year', aggfunc='sum').reset_index()
    sulfur_df.rename(columns={'SO2' : 'SO2_sum'}, inplace = True)
    return sulfur_df
sulfur_df = sum_sulfur(df)

def sum_carbon(df):
    carb_df = df.pivot_table(values = 'CO', index='year', aggfunc='sum').reset_index()
    carb_df.rename(columns={'CO' : 'CO_sum'}, inplace = True)
    return carb_df
carb_df = sum_carbon(df)

def sum_nitrogen(df):
    nit_df = df.pivot_table(values = 'NO2', index='year', aggfunc='sum').reset_index()
    nit_df.rename(columns={'NO2' : 'NO2_sum'}, inplace = True)
    return nit_df
nit_df = sum_nitrogen(df)

def sum_ozone(df):
    ozone_df = df.pivot_table(values = 'O3', index='year', aggfunc='sum').reset_index()
    ozone_df.rename(columns={'O3' : 'O3_sum'}, inplace = True)
    return ozone_df
ozone_df = sum_ozone(df)

col1, col2= st.columns(2)

with col1:
    sum_of_sulfur = sulfur_df['SO2_sum'].sum()
    col1.metric('Total of Sulfur Dioxide In The Air', value = int(sum_of_sulfur))

with col2:
    sum_of_carbon = carb_df['CO_sum'].sum()
    col2.metric('Total of Carbon Monoxide In The Air', value = int(sum_of_carbon))

col3, col4 = st.columns(2)

with col3:
    sum_of_nitrogen = nit_df['NO2_sum'].sum()
    col3.metric('Total of Nitrogen In The Air', value = int(sum_of_nitrogen))

with col4:
    sum_of_ozone = ozone_df['O3_sum'].sum()
    col4.metric('Total of Ozone In The Air', value = int(sum_of_ozone))

# time series plot with trend line
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')
df.dropna(subset=['datetime'], inplace=True)
df['datetime_int'] = df['datetime'].astype('int64') / 1e9 

# fit trend line for PM10
z = np.polyfit(df['datetime_int'], df['PM10'], 1)
p = np.poly1d(z)
df['trendline'] = p(df['datetime_int'])

# dashboard title #1
st.title("PM10 Level Over Time Dashboard")

# interactive filter data based on input
df_filter = df[(df['year'] == input_year) & (df['month'] == input_month)]
if df_filter.empty:
    st.warning("No data available for the selected year and month.")
else:
    # plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_filter['datetime'], df_filter['PM10'], label='Original Data')
    ax.plot(df_filter['datetime'], df_filter['trendline'], color='red', label='Trend Line')
    ax.set_xlabel('Date')
    ax.set_ylabel('PM10 Concentration')
    ax.set_title(f'PM10 Level Over Time - {input_year}-{input_month:02d}')
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# finding the hour with highest average
hourly_pm25 = df.groupby('hour')['PM2.5'].mean()
max_hour = hourly_pm25.idxmax() 
colors = ['skyblue' if hour != max_hour else 'red' for hour in hourly_pm25.index]


# dashboard title #2
st.title('Highest Air Pollutant Time')

# plotting bar chart
plt.figure(figsize=(12, 6))
plt.bar(hourly_pm25.index, hourly_pm25.values, color=colors)
plt.title('Average PM2.5 Levels by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average PM2.5 Levels (µg/m³)')
plt.xticks(range(24)) 
plt.grid(axis='y')

st.pyplot(plt)
st.write(f"The hour with the highest average PM2.5 level is {max_hour+1}:00, with an average of {hourly_pm25[max_hour]:.2f}.")

# dashboard title #3
st.title('Air Quality Comparison Dashboard')

# select box for inputs
year1 = st.selectbox('Select First Year', sorted(df['year'].unique()))
year2 = st.selectbox('Select Second Year', sorted(df['year'].unique()))

# filter data for the selected years
data_year1 = df[df['year'] == year1]
data_year2 = df[df['year'] == year2]

daily_avg_year1 = data_year1.groupby(['year', 'month', 'day'])[['PM10', 'PM2.5', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()
daily_avg_year2 = data_year2.groupby(['year', 'month', 'day'])[['PM10', 'PM2.5', 'SO2', 'NO2', 'CO', 'O3']].mean().reset_index()

# create a date column
daily_avg_year1['date'] = pd.to_datetime(daily_avg_year1[['year', 'month', 'day']])
daily_avg_year2['date'] = pd.to_datetime(daily_avg_year2[['year', 'month', 'day']])

# plotting
plt.figure(figsize=(14, 10))

# plot for PM10
plt.subplot(3, 2, 1)
plt.plot(daily_avg_year1['date'], daily_avg_year1['PM10'], label=str(year1), color='blue')
plt.plot(daily_avg_year2['date'], daily_avg_year2['PM10'], label=str(year2), color='orange')
plt.title('Average PM10 Levels ({} vs {})'.format(year1, year2))
plt.xlabel('Date')
plt.ylabel('PM10 (µg/m³)')
plt.legend()
plt.grid()

# plot for PM2.5
plt.subplot(3, 2, 2)
plt.plot(daily_avg_year1['date'], daily_avg_year1['PM2.5'], label=str(year1), color='blue')
plt.plot(daily_avg_year2['date'], daily_avg_year2['PM2.5'], label=str(year2), color='orange')
plt.title('Average PM2.5 Levels ({} vs {})'.format(year1, year2))
plt.xlabel('Date')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.grid()

plt.tight_layout()
st.pyplot(plt)
