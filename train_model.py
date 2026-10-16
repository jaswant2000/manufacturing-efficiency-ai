from pathlib import Path
import json, time, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ROOT=Path(__file__).resolve().parent
OUT=ROOT/"reports"; OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"Thales_Group_Manufacturing.csv")
df["DateTime"]=pd.to_datetime(df["Date"].astype(str)+" "+df["Timestamp"].astype(str),dayfirst=True,errors="coerce")
df=df.dropna(subset=["DateTime","Efficiency_Status"]).sort_values("DateTime").reset_index(drop=True)
features=["Machine_ID","Operation_Mode","Temperature_C","Vibration_Hz","Power_Consumption_kW","Network_Latency_ms","Packet_Loss_%","Quality_Control_Defect_Rate_%","Production_Speed_units_per_hr","Predictive_Maintenance_Score","Error_Rate_%"]
df["Machine_ID"]=df["Machine_ID"].astype(str)
X,y=df[features],df["Efficiency_Status"]
split=int(len(df)*0.8); X_train,X_test=X.iloc[:split],X.iloc[split:]; y_train,y_test=y.iloc[:split],y.iloc[split:]
categorical=["Machine_ID","Operation_Mode"]; numeric=[c for c in features if c not in categorical]
pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),categorical),("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric)])
models={"Logistic Regression":LogisticRegression(max_iter=1000,class_weight="balanced"),"Random Forest":RandomForestClassifier(n_estimators=60,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=42)}
results={}; fitted={}
labels=["High","Medium","Low"]
for name,estimator in models.items():
    pipe=Pipeline([("preprocess",pre),("model",estimator)])
    start=time.time(); pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
    report=classification_report(y_test,pred,labels=labels,output_dict=True,zero_division=0)
    results[name]={"accuracy":float(accuracy_score(y_test,pred)),"macro_precision":float(report["macro avg"]["precision"]),"macro_recall":float(report["macro avg"]["recall"]),"macro_f1":float(report["macro avg"]["f1-score"]),"weighted_f1":float(report["weighted avg"]["f1-score"]),"per_class":{l:{m:float(report[l][m]) for m in ["precision","recall","f1-score","support"]} for l in labels},"confusion_matrix":confusion_matrix(y_test,pred,labels=labels).tolist(),"training_seconds":round(time.time()-start,2)}
    fitted[name]=pipe
    print(name, "accuracy",round(results[name]["accuracy"],4),"macro-F1",round(results[name]["macro_f1"],4))
best=max(results,key=lambda n:results[n]["macro_f1"])
joblib.dump({"model":fitted[best],"features":features,"classes":list(fitted[best].classes_),"model_name":best},ROOT/"model.joblib")
json.dump({"best_model":best,"results":results,"split":"chronological first 80% train, latest 20% test","features":features},open(OUT/"model_metrics.json","w"),indent=2)
cm=np.array(results[best]["confusion_matrix"])
plt.figure(figsize=(5,4)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels); plt.title("Confusion Matrix — "+best); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close()
model=fitted[best].named_steps["model"]
names=fitted[best].named_steps["preprocess"].get_feature_names_out()
if hasattr(model,"feature_importances_"):
    imp=pd.Series(model.feature_importances_,index=names).sort_values(ascending=False).head(15)
    imp.to_csv(OUT/"feature_importance_top15.csv",header=["importance"])
    plt.figure(figsize=(8,5)); imp.sort_values().plot(kind="barh"); plt.title("Top Feature Importances"); plt.tight_layout(); plt.savefig(OUT/"feature_importance.png",dpi=160); plt.close()
print("Selected by macro-F1:",best)
