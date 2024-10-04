# Air Quality Analysis at Shunyi Station 🚋

## Overview
Air Quality Analysis is one of the 3 dataset that provided by Dicoding, as part of the final project to implemented the learning at "Belajar Analisis Data dengan Python" course. This analysis will provide a comprehensive analysis starts from gathering, assessing, and cleaning data, Exploratory Data Analysis (EDA), and etc.

## Table of Contents
1. [Objective](#objective)
2. [Directories](#directories)
3. [Data](#data)
4. [HOW TO RUN AIR QUALITY DASHBOARD](#how-to-run-air-quality-dashboard)
5. [Contact](#contact)


## Objective
This analysis provide answers to three (3) business questions, specifically by doing analyzing trend in PM10 concentration levels, finding the highest air pollutant time based on PM2.5, and comparing PM2.5 and PM10 in two years perspective.

## Directories
- /data: Directory that contain raw data with .csv format used in this analysis.
- /dashboard: Directory that contains dashboard for the analysis in the format of main.py and the cleaned data.

## Data
The data that used in this analysis could be found [here](https://github.com/marceloreis/HTI/tree/master)

## HOW TO RUN AIR QUALITY DASHBOARD
###Setup Environment

1. **Create Python Environment**
   - Setup Environment - Anaconda
   ```
   conda create --name airquality-ds python=3.9
   conda activate main-ds
   ```
   - Setup Environment - Shell/Terminal
     ```
     mkdir proyek_analisis_data
     cd proyek_analisis_data
     pipenv install
     pipenv shell
     ```
2. **Install Packages**
   - Packages that used in this analysis need to be installed first.
     ```
     pip install -r requirements.txt
     ```
3. **Run the Dashboard**
   - Go to the project directory and run the streamlit app.
     ```
     cd Submission/dashboard/
     streamlit run dashboard.py
     ```
     or visit the website [here](https://ferdiikhsan-air-analysis-shunyi-station.streamlit.app/)

## Contact
If you find any problem with the analysis, please contact me immediately.
bye👋
