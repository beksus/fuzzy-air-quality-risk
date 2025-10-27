import pandas as pd
path = r"C:\Users\xd\PycharmProjects\PythonProject1\kuala-lumpur-air-quality.csv"  # or original CSV
df = pd.read_csv(path)
print(df[['date','pm25','no2','o3']].head(10))
print("\nTypes:\n", df[['pm25','no2','o3']].dtypes)
print("\nStats:\n", df[['pm25','no2','o3']].describe())
print("\nUnique counts (first 20):")
print(df['pm25'].value_counts().head(20))
