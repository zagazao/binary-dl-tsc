import mlflow

mlflow.set_tracking_uri('file:///home/lukas/data/newruns/1/mlruns/')

df = mlflow.search_runs([])

print(df)

metric_cols = [col for col in df.columns if 'metrics' in col]

print(metric_cols)

metrics_mean_df = df[metric_cols].groupby('repeat').mean()
#
df_mean = df.groupby(['tags.dataset', 'tags.classifier_name', 'tags.architecture_idx']).mean()
df_std = df.groupby(['tags.dataset', 'tags.classifier_name', 'tags.architecture_idx']).std()

print(metrics_mean_df)

for dataset_name, sub_df in df.groupby('tags.dataset'):
    ...
