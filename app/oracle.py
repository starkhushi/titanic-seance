"""The numbers side of the seance: TabPFN's survival table and the real passenger manifest.

TabPFN ran once on Kaggle over every profile the form can produce (see kaggle/tabpfn_grid),
so the server only looks numbers up. Nothing here calls a model, and Gemma is only ever
handed what this module computed.
"""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
PORT_NAMES = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
CLASS_NAMES = {1: "first class", 2: "second class", 3: "third class"}


@lru_cache(maxsize=1)
def grid() -> dict:
    with open(DATA / "survival_grid.json", encoding="utf-8") as f:
        g = json.load(f)
    sizes = [len(g["axes"][k]) for k in g["features"]]
    strides, s = [], 1
    for n in reversed(sizes):
        strides.append(s)
        s *= n
    g["_strides"] = list(reversed(strides))
    return g


def clean_profile(p: dict) -> dict:
    def num(k, lo, hi, default):
        try:
            return min(hi, max(lo, int(round(float(p.get(k, default))))))
        except (TypeError, ValueError):
            return default

    port = str(p.get("port", "S")).upper()[:1]
    return dict(pclass=num("pclass", 1, 3, 3), female=bool(p.get("female", False)),
                age=num("age", 0, 80, 30), sibsp=num("sibsp", 0, 4, 0), parch=num("parch", 0, 3, 0),
                port=port if port in PORT_NAMES else "S")


def survival(p: dict) -> float:
    g = grid()
    values = dict(Pclass=p["pclass"], Sex=int(p["female"]), Age=p["age"], SibSp=p["sibsp"],
                  Parch=p["parch"], Embarked=g["ports"].index(p["port"]))
    idx = sum(g["axes"][k].index(values[k]) * st for k, st in zip(g["features"], g["_strides"]))
    return g["probs"][idx]


def what_ifs(p: dict) -> list[dict]:
    """Single changes to the profile and what TabPFN says about each, biggest swing first."""
    base = survival(p)
    alts = []
    for c in (1, 2, 3):
        if c != p["pclass"]:
            alts.append((f"travelling {CLASS_NAMES[c]}", dict(p, pclass=c)))
    alts.append(("recorded as a man" if p["female"] else "recorded as a woman", dict(p, female=not p["female"])))
    if p["sibsp"] or p["parch"]:
        alts.append(("travelling alone", dict(p, sibsp=0, parch=0)))
    else:
        alts.append(("travelling with a spouse", dict(p, sibsp=1)))
    if p["age"] >= 16:
        alts.append(("a child of 8", dict(p, age=8)))
    for port in PORT_NAMES:
        if port != p["port"]:
            alts.append((f"boarding at {PORT_NAMES[port]}", dict(p, port=port)))
    out = [dict(change=label, prob=survival(q), delta=survival(q) - base) for label, q in alts]
    return sorted(out, key=lambda r: -abs(r["delta"]))


@lru_cache(maxsize=1)
def manifest() -> list[dict]:
    rows = []
    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                id=int(r["PassengerId"]), name=display_name(r["Name"]), pclass=int(r["Pclass"]),
                female=r["Sex"] == "female", age=float(r["Age"]) if r["Age"] else None,
                sibsp=int(r["SibSp"]), parch=int(r["Parch"]), port=r["Embarked"] or None,
                ticket=r["Ticket"], fare=round(float(r["Fare"]), 2), cabin=r["Cabin"] or None,
                survived=r["Survived"] == "1"))
    return rows


def display_name(raw: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr. Owen Harris Braund' (manifest spelling kept)."""
    surname, _, rest = raw.partition(",")
    rest = re.sub(r"\s+", " ", rest).strip()
    return f"{rest} {surname.strip()}" if rest else surname.strip()


def kindred(p: dict, k: int = 3) -> list[dict]:
    """The real passengers whose manifest lines look most like the visitor's."""
    def dist(r):
        d = 0.0 if r["pclass"] == p["pclass"] else 6.0
        d += 0.0 if r["female"] == p["female"] else 8.0
        d += abs(r["age"] - p["age"]) / 4 if r["age"] is not None else 4.0
        d += abs(min(r["sibsp"], 4) - p["sibsp"]) + abs(min(r["parch"], 3) - p["parch"])
        d += 0.0 if r["port"] == p["port"] else 1.5
        return d

    return [dict(r, port_name=PORT_NAMES.get(r["port"] or "", "unknown"))
            for r in sorted(manifest(), key=lambda r: (dist(r), r["id"]))[:k]]


def reading(raw: dict) -> dict:
    p = clean_profile(raw)
    g = grid()
    return dict(profile=p, prob=survival(p), what_ifs=what_ifs(p), souls=kindred(p),
                base_rate=g["base_rate"], cv=g["cv"], model=g["model"], train_rows=g["train_rows"])
