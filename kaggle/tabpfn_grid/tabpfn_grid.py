# Titanic Seance: fit TabPFN on the 891 known passengers, check it with 5-fold CV,
# then score every passenger profile the app lets you pick and save it as a lookup table.
import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=False)

import numpy as np
import pandas as pd
import torch
import tabpfn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

print("tabpfn", getattr(tabpfn, "__version__", "?"), "torch", torch.__version__, "cuda", torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working"

train_path = glob.glob("/kaggle/input/**/train.csv", recursive=True)
train_path = [p for p in train_path if "titanic" in p.lower()] or train_path
root = os.path.dirname(train_path[0])
train = pd.read_csv(f"{root}/train.csv")
test = pd.read_csv(f"{root}/test.csv")
print("data", root, train.shape, test.shape)

PORTS = ["S", "C", "Q"]          # Southampton, Cherbourg, Queenstown
FEATS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]


def encode(df):
    x = pd.DataFrame({
        "Pclass": df["Pclass"].astype(float),
        "Sex": (df["Sex"] == "female").astype(float),
        "Age": df["Age"].astype(float),                 # TabPFN takes the NaNs as they are
        "SibSp": df["SibSp"].clip(upper=4).astype(float),
        "Parch": df["Parch"].clip(upper=3).astype(float),
        "Embarked": df["Embarked"].fillna("S").map({p: i for i, p in enumerate(PORTS)}).astype(float),
    })
    return x[FEATS].to_numpy()


X, y = encode(train), train["Survived"].to_numpy()
CAT = [0, 1, 5]


def make_model():
    # Newer tabpfn releases default to gated v2.5 weights; the plain v2 weights download without a token.
    try:
        from tabpfn.constants import ModelVersion
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device=DEVICE)
        m.set_params(categorical_features_indices=CAT)
        return m
    except Exception as e:  # noqa: BLE001
        print("create_default_for_version unavailable:", repr(e))
        return TabPFNClassifier(device=DEVICE, categorical_features_indices=CAT)


# ---- honest check: 5-fold CV against two simple baselines
skf = StratifiedKFold(5, shuffle=True, random_state=0)
oof = np.zeros(len(y))
oof_lr = np.zeros(len(y))
t0 = time.time()
for tr, va in skf.split(X, y):
    m = make_model().fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
    Xl = np.nan_to_num(X, nan=np.nanmedian(X[tr][:, 2]))
    lr = LogisticRegression(max_iter=2000).fit(Xl[tr], y[tr])
    oof_lr[va] = lr.predict_proba(Xl[va])[:, 1]
cv = dict(
    tabpfn=dict(acc=accuracy_score(y, oof > 0.5), auc=roc_auc_score(y, oof), logloss=log_loss(y, oof)),
    logistic=dict(acc=accuracy_score(y, oof_lr > 0.5), auc=roc_auc_score(y, oof_lr), logloss=log_loss(y, oof_lr)),
    women_survive=dict(acc=accuracy_score(y, X[:, 1] == 1)),
    seconds=round(time.time() - t0, 1),
)
print(json.dumps(cv, indent=1))

# ---- final model on all 891 passengers
model = make_model().fit(X, y)
pd.DataFrame({"PassengerId": test["PassengerId"],
              "Survived": (model.predict_proba(encode(test))[:, 1] > 0.5).astype(int)}
             ).to_csv(f"{OUT}/submission.csv", index=False)

# ---- every profile the app offers: class x sex x age 0..80 x sibsp 0..4 x parch 0..3 x port
AXES = dict(Pclass=[1, 2, 3], Sex=[0, 1], Age=list(range(0, 81)), SibSp=list(range(5)),
            Parch=list(range(4)), Embarked=[0, 1, 2])
grid = np.array(np.meshgrid(*AXES.values(), indexing="ij"), dtype=float).reshape(len(AXES), -1).T
t0 = time.time()
probs = np.concatenate([model.predict_proba(grid[i:i + 5000])[:, 1] for i in range(0, len(grid), 5000)])
print("grid", grid.shape, "scored in", round(time.time() - t0, 1), "s")

table = dict(
    model=f"TabPFN v2 (tabpfn {getattr(tabpfn, '__version__', '?')})",
    features=FEATS, ports=PORTS, axes=AXES,
    note="probs is row-major over axes in the order of features (C order)",
    probs=[round(float(p), 4) for p in probs],
    cv=cv, train_rows=int(len(y)), base_rate=float(y.mean()),
)
with open(f"{OUT}/survival_grid.json", "w") as f:
    json.dump(table, f, separators=(",", ":"))

# a few sanity rows
for desc, row in [("1st class woman, 30, alone, S", [1, 1, 30, 0, 0, 0]),
                  ("3rd class man, 25, alone, S", [3, 0, 25, 0, 0, 0]),
                  ("3rd class boy, 6, 3 siblings, 2 parents, S", [3, 0, 6, 3, 2, 0]),
                  ("2nd class girl, 4, with parents, C", [2, 1, 4, 1, 2, 1])]:
    print(desc, round(float(model.predict_proba(np.array([row], dtype=float))[0, 1]), 3))
print("done")
