# Pneumonia Chest X-ray Classification: ML Pipeline

African Leadership University, BSE. Machine Learning Pipeline (summative assignment).

An end-to-end machine learning pipeline built on image data instead of tables. A CNN
classifies pediatric chest X-rays as `NORMAL` or `PNEUMONIA`. It runs behind a FastAPI service
with a web dashboard, ships in Docker, deploys to the cloud, and gets load-tested with Locust.
You can also retrain it from the browser by uploading new images and clicking a button.

---

## Links

| | |
|---|---|
| Video demo (YouTube) | `<ADD YOUR YOUTUBE LINK HERE>` |
| Live URL | https://pneumonia-xray-classifier-tqu5.onrender.com |
| GitHub repo | https://github.com/supserrr/pneumonia-detection-pipeline |

> Add your YouTube demo link above before submitting; the live URL and repo are set.
> The live URL runs on Render's free tier and sleeps after inactivity, so the first
> request can take 30 to 60 seconds to wake the instance.

---

## Project description

Chest X-rays are the cheapest and most widely available lung imaging test, but reading one still
takes a trained radiologist. This project trains a convolutional neural network to tell pneumonic
chest X-rays from healthy ones. More to the point of the assignment, it wraps that model in a full
production lifecycle: acquisition, preprocessing, training, evaluation, deployment, monitoring, load
testing, and a retraining loop that lets a user push new data into the model without touching code.

The dataset is Kaggle's [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
(Kermany et al.): chest X-rays of pediatric patients aged 1 to 5 from Guangzhou Women and Children's
Medical Center. All 5,840 images ship with this repo, converted to 64×64 grayscale PNG:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Test  | 234 | 390 | 624 |

The dataset's own `val/` split holds only 16 images, too few to be useful, so I discard it and carve
a stratified validation split out of train instead.

The model is a compact CNN (about 30k parameters) trained with Adam, data augmentation, L2
regularization, batch normalization, dropout, global average pooling, early stopping,
ReduceLROnPlateau, and class weights.

### Results on the held-out 624-image test set

| Metric | Score |
|---|---|
| Accuracy | 0.8622 |
| Precision | 0.8568 |
| Recall | 0.9359 |
| F1 score | 0.8946 |
| ROC AUC | 0.9358 |

Confusion matrix `[[173, 61], [25, 365]]`: 25 missed pneumonia cases and 61 false alarms.

The test set is 62.5% pneumonia, so a majority-class baseline already scores 0.625. That, not 50%,
is the bar worth comparing against, and 0.862 clears it by enough to be a real improvement rather
than an artefact of the imbalance.

A few honest caveats. The data is entirely pediatric (ages 1 to 5), so none of it transfers to adult
chest X-rays. Downsampling to 64×64 throws away detail a radiologist would use. The test set is a
single curated split. This is a pipeline-engineering project, not a diagnostic tool.

---

## The most important finding: a train/test distribution shift

The first model I trained here, with no augmentation, scored 98.5% on validation but 73.1% on the
test set. That gap is not a bug. It is a property of this dataset, and catching it mattered more than
any hyperparameter I tuned.

Here is the diagnostic. On NORMAL images the model gave a mean pneumonia probability of 0.030 on
validation but 0.685 on the test set. Validation is carved out of *train*, so it shares train's
distribution, while Kaggle's official `test/` split was curated separately and genuinely differs. The
model had memorized train-specific quirks and fell apart on real held-out data, even as validation
cheerfully reported 98.5%.

The fix was data augmentation (rotation, zoom, translation, contrast), which pushes the model toward
features that survive small perturbations:

| | Test accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Without augmentation | 0.7308 | 0.6989 | 1.0000 | 0.8228 | 0.9234 |
| With augmentation | 0.8622 | 0.8568 | 0.9359 | 0.8946 | 0.9358 |

Look at what the un-augmented model actually did. Recall of 1.0 at precision 0.70 means it flagged
168 of the 234 healthy children as pneumonic. It caught every real case by calling almost everyone
sick, which is a rubber stamp rather than a model. Accuracy alone would never have shown this. The
confusion matrix did.

The lesson: a validation score drawn from the training distribution can be silently and badly
optimistic. The full reasoning is in the notebook.

---

## Repository structure

```
pneumonia_detection_pipeline/
├── README.md
├── notebook/
│   └── pneumonia_classification.ipynb    # preprocessing, training, evaluation, prediction
├── src/
│   ├── preprocessing.py                  # data acquisition + processing + dataset stats
│   ├── model.py                          # build / train / evaluate / save / RETRAIN
│   └── prediction.py                     # single-image prediction (backs POST /predict)
├── app/
│   ├── main.py                           # FastAPI: /predict /upload /retrain /status /visualizations
│   └── static/index.html                 # web dashboard UI
├── locust/
│   ├── locustfile.py                     # flood request simulation
│   └── sample_xray.png
├── data/
│   ├── train/{NORMAL,PNEUMONIA}/         # 5,216 images
│   └── test/{NORMAL,PNEUMONIA}/          # 624 images
├── models/
│   ├── pneumonia_model.h5                # trained model
│   └── model_metadata.json               # version + live metrics (shown in the UI)
├── results/
│   ├── flood_simulation_results.md       # Locust write-up
│   ├── locust_{1_worker,2_workers,4_workers}_stats.csv
│   ├── test_metrics.json
│   ├── threshold_sweep.json
│   └── training_history.json
├── Dockerfile
├── docker-compose.yml                    # nginx + scalable api replicas
├── nginx.conf
├── render.yaml                           # cloud deployment blueprint
├── requirements.txt                      # runtime deps (what Docker installs)
└── requirements-dev.txt                  # + notebook & Locust deps
```

---

## Setup

### 1. Run locally with Python

```bash
git clone <YOUR_REPO_URL>
cd pneumonia_detection_pipeline

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt   # runtime only, enough to serve the API

uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 for the dashboard, or http://localhost:8000/docs for the Swagger docs.

The trained model ships with the repo, so it predicts right away with no training needed.

> Note: `tensorflow==2.15` has wheels for Python 3.9 to 3.11. On Python 3.12 or 3.13 the install
> will fail, so use a 3.10 or 3.11 interpreter for the venv.

### 2. Run the notebook

The notebook needs a few extra packages (matplotlib, pandas, scipy, jupyter) that the API does not,
so install the dev requirements:

```bash
pip install -r requirements-dev.txt
jupyter notebook notebook/pneumonia_classification.ipynb
```

### 3. Re-acquire the data from Kaggle (optional)

The processed data already ships with the repo. To re-download from scratch you need Kaggle API
credentials at `~/.kaggle/kaggle.json` (Kaggle → Account → Create New API Token):

```bash
pip install kaggle
python -c "from src.preprocessing import acquire_data; acquire_data(force=True)"
```

### 4. Retrain from scratch (optional)

```bash
python -m src.model        # trains, evaluates, saves the .h5  (~4 s/epoch on 4 CPU cores)
```

### 5. Run with Docker

```bash
docker build -t pneumonia-xray .
docker run -p 8000:8000 pneumonia-xray
```

### 6. Run multiple containers behind a load balancer

```bash
docker compose up --build --scale api=2     # nginx balances on http://localhost:8080
```

---

## Deploying to the cloud (Render)

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com), then New → Blueprint, and connect the repo. Render reads
   `render.yaml` and builds the `Dockerfile` for you.
   *(Manual alternative: New → Web Service, runtime Docker, health check path `/status`.)*
3. Wait for the build, then open the service URL. The dashboard is at `/`.

Evaluating the model in production: `GET /status` returns live uptime plus the metrics and version
note of the model currently loaded in the container.

```json
{
  "status": "up",
  "uptime_human": "0d 2h 14m 07s",
  "model_saved_at": "2026-07-16 03:11:42",
  "model_note": "initial training - 21 epochs w/ augmentation on 5,216 pediatric chest X-rays (64x64)",
  "model_metrics": {"accuracy": 0.8622, "precision": 0.8568, "recall": 0.9359,
                    "f1_score": 0.8946, "roc_auc": 0.9358, "confusion_matrix": [[173,61],[25,365]]},
  "retraining": "idle"
}
```

Every retrain re-evaluates against the held-out test set and rewrites these numbers, so the dashboard
always shows the real quality of the model currently serving traffic. If a retrain makes things
worse, that shows up immediately.

---

## Using the app

| Feature | How |
|---|---|
| Predict one X-ray | Dashboard → *Predict* → choose an image → **Predict**. Returns class + probability + confidence. Try any file from `data/test/`. |
| Visualizations | Dashboard → *Dataset Visualizations*: class balance, brightness, contrast, and the mean image per class, each with a written interpretation. |
| Upload bulk data | Dashboard → *Upload bulk data* → pick a label → select many images → **Upload**. |
| Trigger retraining | Dashboard → **🔁 Retrain model**. Fine-tunes the saved model on base + uploaded data, re-evaluates, hot-swaps it into the live API. |
| Monitor up-time | Dashboard → *Model Up-time & Health*, refreshed every 5 s. |

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | web dashboard |
| `GET` | `/status` | up-time, model version, live metrics |
| `GET` | `/visualizations` | dataset statistics powering the charts |
| `POST` | `/predict` | classify one uploaded image |
| `POST` | `/upload` | bulk-upload labelled images (`files`, `label`) |
| `POST` | `/retrain` | trigger retraining (runs in the background) |
| `GET` | `/retrain/status` | progress/result of the last retrain |

```bash
curl -F "file=@data/test/PNEUMONIA/person100_bacteria_475.png" http://localhost:8000/predict
# {"prediction":"PNEUMONIA","pneumonia_probability":0.9746,"confidence":0.9746,...}
```

### How retraining works

The trigger does not train from zero. `retrain_model()` loads `models/pneumonia_model.h5` as a
pre-trained base, joins the original 5,216 training images with everything uploaded to
`data/uploads/`, fine-tunes at a reduced learning rate (1e-4), re-evaluates on the untouched test
set, saves the new model with an updated version note, hot-swaps it into the running API, and
archives the consumed uploads so they never get trained on twice.

---

## Results from the flood request simulation (Locust)

50 concurrent users, spawn rate 25/s, about 15 s per run, a real X-ray posted to `/predict`, on a
4 vCPU / 3.9 GB host.

| Replicas | Requests | Failures | Avg (ms) | Median (ms) | p95 (ms) | Max (ms) | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 worker  | 529 | 0 (0.00%) | 134 | 90 | 490 | 748 | 35.71 |
| 2 workers | 544 | 0 (0.00%) | 62 | 36 | 280 | 353 | 39.21 |
| 4 workers | 511 | 0 (0.00%) | 53 | 30 | 250 | 367 | 39.66 |

No requests failed at any scale. Under flood the service slows down but never returns errors. Going
from 1 to 2 replicas cut median latency about 2.5× (90 ms to 36 ms). Going from 2 to 4 bought very
little (36 ms to 30 ms), which is what you would expect once the replica count reaches the 4 physical
cores.

Read the latency, not the throughput. With 50 users and a mean wait of 1.25 s, Locust cannot offer
much more than 40 req/s, so every config with 2 or more workers lands around 38 to 40. That number
measures the load generator's ceiling, not the server's. Concluding "the API maxes out at 40 req/s"
from this data would be wrong. Latency is the signal that still discriminates.

One correction, kept on the record. An earlier run of the same 4-worker config reported 25 req/s with
a 420 ms median, worse than a single worker, and I first wrote it up as evidence that over-scaling
backfires. It did not reproduce. Two repeat runs gave medians of 30 ms and 28 ms. The original was
almost certainly an artefact of memory pressure right after a training run. A single 15-second run on
a shared host is noisy, and one measurement is not a result, so I left the anomaly documented instead
of quietly swapping in the tidier number.

Full write-up and raw CSVs: [`results/flood_simulation_results.md`](results/flood_simulation_results.md).

> Methodology note. Docker was not available where these numbers were measured, so each replica here
> is a uvicorn worker process (`--workers N`). Each one is an independent process with its own
> TensorFlow runtime and model copy, competing for the same CPUs the way containers on one host
> would. To reproduce with real containers, run `docker compose up --build --scale api=2` and point
> Locust at `http://localhost:8080`.

---

## Bugs worth documenting

All three came from distrusting a convenient number. Each is written up in the notebook.

1. Non-stratified validation split. Keras' `validation_split` takes the *last* fraction of the arrays
   without shuffling. The data is loaded class by class, so the validation set came out 100% one
   class and `val_accuracy` was a meaningless 0.0 or 1.0. Fixed with an explicit stratified split.
2. BatchNorm moving statistics never converged. At Keras' default momentum of 0.99 with few steps per
   epoch, the moving mean and variance barely moved from their initial values, so the model scored
   97% in training mode and collapsed to 50% at inference. Fixed by lowering BN momentum to 0.9.
3. Train/test distribution shift, described above. Fixed with data augmentation.

---

## Tech stack

TensorFlow/Keras · FastAPI · Uvicorn · scikit-learn · SciPy · Pillow · NumPy · Locust ·
Docker · nginx · Render. The dashboard is dependency-free vanilla JS with hand-rolled, animated SVG
charts (no chart library, no external fonts or CDNs), so it loads instantly and is theme-aware
(light/dark).

## Data attribution

Kermany, D., Zhang, K., Goldbaum, M. (2018), *Labeled Optical Coherence Tomography (OCT) and Chest
X-Ray Images for Classification*, Mendeley Data, distributed via
[Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) under CC BY 4.0.
Used here for education, not for clinical use.
