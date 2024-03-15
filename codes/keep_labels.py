import pandas as pd

df = pd.read_csv('pre_proccessed.csv')
desiered_labels = ['neutral', 'sadness', 'worry', 'happiness']
filtered_df = df[df['emotions'].isin(desiered_labels)]
filtered_df.to_csv('4_emotions_data.csv')