from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"reports"; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"Thales_Group_Manufacturing.csv")
df["DateTime"]=pd.to_datetime(df["Date"].astype(str)+" "+df["Timestamp"].astype(str),dayfirst=True,errors="coerce")
summary={"rows":int(len(df)),"source_columns":int(len(df.columns)),"missing_cells":int(df.isna().sum().sum()),"duplicate_rows":int(df.duplicated().sum()),"invalid_datetime_rows":int(df["DateTime"].isna().sum()),"date_min":str(df["DateTime"].min()),"date_max":str(df["DateTime"].max()),"unique_machines":int(df["Machine_ID"].nunique()),"operation_mode_counts":df["Operation_Mode"].value_counts().to_dict(),"target_counts":df["Efficiency_Status"].value_counts().to_dict(),"target_percent":(df["Efficiency_Status"].value_counts(normalize=True)*100).round(2).to_dict(),"numeric_describe":df.select_dtypes(include=np.number).describe().round(3).to_dict()}
json.dump(summary,open(OUT/"eda_summary.json","w"),indent=2)
print(json.dumps({k:v for k,v in summary.items() if k!="numeric_describe"},indent=2))
counts=df["Efficiency_Status"].value_counts().reindex(["Low","Medium","High"],fill_value=0)
plt.figure(figsize=(7,4)); counts.plot(kind="bar"); plt.title("Efficiency Status Distribution"); plt.ylabel("Records"); plt.tight_layout(); plt.savefig(OUT/"target_distribution.png",dpi=160); plt.close()
num=df.select_dtypes(include=np.number)
plt.figure(figsize=(10,7)); sns.heatmap(num.corr(),cmap="coolwarm",center=0); plt.title("Correlation Heatmap"); plt.tight_layout(); plt.savefig(OUT/"correlation_heatmap.png",dpi=160); plt.close()
print(df.groupby("Efficiency_Status").mean(numeric_only=True).round(3))
