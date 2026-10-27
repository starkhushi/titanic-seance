# Titanic Séance

Tell the spirits who you would have been in April 1912: your ticket class, your age, who you travelled with, and where you boarded. TabPFN, trained on the 891 passengers whose fate is recorded, gives your chance of surviving the night. Then the real passenger whose manifest line is closest to yours speaks to you, through Gemma 4.

Built for the DEV Hacktoberfest 2026 "Ghost in the Machine" challenge.

## How it works

```
your profile ──► TabPFN lookup table (29,160 profiles, scored once on Kaggle)
             ──► what-ifs (one thing changed at a time)
             ──► nearest real passengers in the manifest
                         │
                         ▼
             Gemma 4 speaks as the closest passenger,
             using only those numbers and that manifest line
```

- **TabPFN-3.5** ([kaggle/tabpfn_grid](kaggle/tabpfn_grid/tabpfn_grid.py)), Prior Labs' open-weight tabular foundation model, is fit on the Kaggle Titanic training set (class, sex, age, siblings/spouse, parents/children, port) on a Kaggle T4. It scores every profile the form can make, so the web server doesn't need torch and fits in Render's free 512 MB. The kernel compares 3.5 against v2 and keeps the one with the better CV log loss.
- **Python decides every fact** ([app/oracle.py](app/oracle.py)): the survival chance, the what-ifs, and which passengers are closest to you.
- **Gemma 4** ([app/ghost.py](app/ghost.py)) only narrates. These are real people, so the prompt gives it their manifest line and a short list of well-known facts about the sinking, and tells it not to invent a hometown, a job or a family for them.

5-fold cross-validation on the 891 passengers:

| model | accuracy | AUC | log loss |
|---|---|---|---|
| TabPFN-3.5 (used) | 0.825 | 0.857 | 0.416 |
| TabPFN v2 | 0.832 | 0.857 | 0.428 |
| logistic regression | 0.790 | 0.843 | 0.459 |
| "women survive" rule | 0.787 | – | – |

3.5 gives better-calibrated probabilities (lower log loss). That matters more here than accuracy, because the page shows a percentage.

To rerun the kernel with 3.5 you need a Prior Labs key, with the TabPFN-3.5 license accepted at [ux.priorlabs.ai](https://ux.priorlabs.ai). The kernel reads it from a private dataset as `TABPFN_TOKEN`. The 3.5 weights are under a non-commercial license, and so is the survival table they produced; this project is a non-commercial demo.

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # free key from https://aistudio.google.com/apikey
uvicorn app.main:app --reload
```

To run Gemma yourself instead (Ollama, llama.cpp `llama-server`, or any OpenAI-compatible server):

```bash
export LLM_BACKEND=openai LLM_BASE_URL=http://127.0.0.1:11434/v1 LLM_MODEL=gemma3:4b
```

Without a key, the page still works: the ghost falls back to a plain template built from the same facts.

Shareable links fill in the form and start the séance: `/?class=3&sex=m&age=25&sibsp=0&parch=0&port=S&lang=en`.

## Deploy

[render.yaml](render.yaml) describes a free Render web service. Set `GEMINI_API_KEY` in the dashboard.

## Data

`data/manifest.csv` is the Kaggle Titanic training set (891 passengers, originally compiled from Encyclopedia Titanica). `data/survival_grid.json` is the output of the Kaggle kernel above.

## License

MIT
