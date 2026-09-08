# Cross-Vehicle Generalisation of In-Vehicle Intrusion Detection

**A controlled benchmark over eight vehicles and three public CAN corpora**

Does a CAN-bus intrusion detector built on one vehicle work on a vehicle it has
never seen? Almost every published detector is trained and tested on the same
car and reports near-perfect scores. This project measures what happens when
the car changes, separates the cost of the change from the effects of training
volume and fleet diversity, and shows that the input representation, not the
learning algorithm, decides the outcome.

Companion documents in `paper/`:

| File | Audience |
|---|---|
| `Experiment1_CrossVehicle_Paper_byFable_v2.docx` | Academic paper (with table of contents, related-work comparison, 50 references) |
| `Experiment1_PlainLanguage_Summary_byFable_v2.docx` | Plain-language explanation for a non-technical reader |
| `research1-Fable vs research1-non-fable.docx` | Comparison with an independent cross-dataset study of the same question |

---

## 1. Background and motivation

The controller area network (CAN) connects the electronic control units of a
vehicle over a shared bus with no authentication and no sender identity. Any
node with bus access can transmit any message identifier, so an attacker who
reaches the bus can flood it, spoof gauge readings, replay commands, silence a
component or abuse the diagnostic channel. Intrusion detection is the practical
mitigation, and a large literature of learned detectors exists.

The literature has a measurement problem. Detectors are evaluated on random or
temporal splits of a single vehicle's traffic and reported with accuracy or
best-threshold F1, under which almost everything scores above 0.99. Whether the
detector works on the next model year, or on a different model in the same
range, is rarely asked. Yet a manufacturer protects a range, not a car, and
message identifiers, signal encodings, bus loads and periods all differ between
vehicles. A detector whose features are built from one vehicle's identifier
vocabulary has, quite literally, no meaningful input on another.

## 2. Aims and objectives

**Aim.** Measure, under a controlled protocol, how well CAN intrusion detectors
generalise to a vehicle from which they have seen no labelled attack, and
determine what governs the outcome.

**Objectives.**

1. Assemble a multi-vehicle benchmark from independently produced public
   corpora with a unified frame schema, a unified attack taxonomy and
   reconstructed per-message labels.
2. Evaluate a broad detector zoo (deterministic rules, an unsupervised learner,
   classical and gradient-boosted learners, four neural architectures, three
   ensembles) under leave-one-vehicle-out with thresholds set only on the
   target vehicle's attack-free traffic at a stated false-alarm budget.
3. Compare two input representations on identical windows: the conventional
   identifier-coupled encoding and an identity-free encoding whose only
   vehicle-specific input is a profile estimated from benign traffic.
4. Separate the cost of changing vehicle from the effects of training volume
   and fleet diversity with explicit controls.
5. Report per-attack-family and per-event recall, include non-AI baselines as
   the benchmark floor, and attach uncertainty to every headline number.

## 3. Corpus

Eight vehicle domains from three independent public datasets. Attack families
are mapped onto one taxonomy (dos, fuzzing, spoofing, replay, diagnostic,
suspension, masquerade, timing). 217 captures, 91.8 million messages, 13.4 hours.

| Domain | Source | Vehicle | Captures | Messages (M) | Hours | Attack-labelled msgs |
|---|---|---|---|---|---|---|
| ROAD_ORNL | ROAD (Oak Ridge National Laboratory) | undisclosed sedan on dynamometer | 45 | 28.2 | 3.45 | 62,513 |
| CIDv2_OpelAstra | CIDv2 (TU Eindhoven) | Opel Astra, urban driving | 8 | 7.6 | 1.08 | 40,073 |
| CIDv2_RenaultClio | CIDv2 | Renault Clio, urban driving | 8 | 1.1 | 0.21 | 40,062 |
| CIDv2_Prototype | CIDv2 | bench prototype | 8 | 0.7 | 0.22 | 45,839 |
| CTT_Impala | can-train-and-test | 2011 Chevrolet Impala | 32 | 11.8 | 2.29 | 132,117 |
| CTT_Silverado | can-train-and-test | 2016 Chevrolet Silverado | 38 | 13.8 | 1.46 | 318,514 |
| CTT_Traverse | can-train-and-test | 2011 Chevrolet Traverse | 36 | 14.2 | 2.01 | 261,999 |
| CTT_Forester | can-train-and-test | 2017 Subaru Forester | 42 | 14.3 | 2.64 | 273,590 |

Data-handling details that matter:

- **can-train-and-test sets mix two vehicles.** Each `set_0X` contains a known
  vehicle in some subdirectories and an unknown vehicle in others. Files are
  mapped to the real vehicle and each vehicle is drawn from exactly one set.
- **Large captures are sliced around the attack**, not truncated to a prefix;
  a prefix cap removes the attack from most captures.
- **CIDv2 labels are recovered by a count-aware anti-join** against the benign
  original, because the documented rules also match legitimate traffic. One
  corrupted log line that the anti-join flagged as an injected frame is
  relabelled benign at load time.
- **Suspension attacks** delete frames, so they carry no positive per-message
  label and are labelled at window level from the documented interval.
- ROAD's accelerator captures have no per-message ground truth and are excluded.

## 4. Method

### 4.1 Windows and profiles

Detection operates on non-overlapping windows of 64 messages. Each vehicle has
a `BenignProfile` estimated from attack-free captures only: per-identifier
median period and interquartile range, modal data length, identifier
vocabulary, nominal bus rate and the set of periodic identifiers.

### 4.2 Two feature spaces

| | FS-A, identifier-coupled (conventional) | FS-B, identity-free (proposed) |
|---|---|---|
| Content | Histogram over the 48 most common identifiers of the *training* traffic plus overflow, raw identifier moments, timing and length statistics, payload-byte means and standard deviations | 40 dimensionless statistics: timing regularity, identifier-distribution shape, conformance to the benign profile (period ratios, unseen identifiers, staleness of periodic identifiers), payload change dynamics, length statistics |
| Dimensions | 76 | 40 |
| Vehicle-specific input | The identifier vocabulary | The `BenignProfile` only |
| Ablations | FS-A-rank: columns aligned by frequency rank rather than identifier | FS-B34: without the six staleness and normalisation features |

### 4.3 Detectors (32 configurations)

- **Rules** (fitted on benign windows only, run both fleet-fitted and fitted on
  the target's own benign traffic): periodicity, unseen identifier, bus load,
  identifier entropy, payload Hamming distance, staleness, a robust-z union and
  a quantile-normalised union.
- **Unsupervised:** isolation forest, both scopes.
- **Supervised:** logistic regression, random forest, extra trees, histogram
  gradient boosting, XGBoost, LightGBM, CatBoost, MLP, 1D-CNN, LSTM,
  Transformer encoder (the last three on real sequences of eight consecutive
  windows of the same capture).
- **Ensembles:** rank-averaged trees, rank-averaged trees plus networks,
  out-of-fold stacking.

### 4.4 Protocol

Every capture's windows are quartered by time, separately for attack and benign
windows, so that each attack event contributes half its windows to each side
of a split.

- **Cross-vehicle (leave-one-vehicle-out):** train on all windows of the other
  seven vehicles (stratified to at most 75k attack and 75k benign windows);
  calibrate thresholds on the target's benign windows of Q1 to Q2; evaluate on
  Q3 to Q4 plus every attack window. The detector never sees a labelled attack
  from the target.
- **In-domain (two folds):** train on Q1 to Q2, calibrate on benign Q3, evaluate
  on Q4 plus attack Q3; then the mirror image. Every attack family is on both
  sides of every fold.
- **Control A, volume-matched:** cross-vehicle training subsampled to exactly
  the in-domain training size and class ratio, evaluated on identical windows.
- **Control B, fleet-size sweep:** k = 1 to 7 training vehicles at fixed volume.
- **Control C, feature ablation:** FS-A, FS-A-rank, FS-B34, FS-B.

### 4.5 Metrics

PR-AUC and ROC-AUC (threshold-free). Recall at 1, 10 and 100 false alarms per
driving hour, threshold set on benign windows only, each budget flagged for
whether the calibration set can resolve it (10 per hour is primary; 1 per hour
needs more than 100,000 benign windows and is resolvable on only three
vehicles). Recall at fixed benign false-positive rates of 1e-3 and 1e-2.
Per-family recall and per-capture event recall at every operating point.
Bootstrap intervals over held-out vehicles and over captures; paired Wilcoxon
tests for every comparison turned into a claim. Accuracy and best-threshold F1
are not reported.

## 5. Results

All numbers are means over the eight held-out vehicles; full tables are in
`results/`.

### 5.1 Headline

| | FS-B (identity-free) | FS-A (identifier-coupled) |
|---|---|---|
| Best supervised detector, cross-vehicle PR-AUC | **0.886** (LightGBM; HistGB 0.886, ensembles 0.882) | 0.638 (rank-average hybrid ensemble) |
| Same detector, in-domain PR-AUC | 0.979 | 0.983 |
| Recall at 10 false alarms/hour, cross-vehicle | 0.75 | 0.48 |
| Best rule, cross-vehicle PR-AUC | 0.819 (robust-z union); 0.805 target-fitted | not applicable |
| Isolation forest, target-fitted | 0.796 | 0.666 |

### 5.2 Findings

1. **Changing vehicle has a real cost.** In the identity-free space the best
   detector loses 0.099 PR-AUC from in-domain to cross-vehicle, and in-domain is
   higher in 74% of detector-vehicle cells (Wilcoxon p < 1e-4). Under the
   conventional encoding the loss is 0.41.
2. **Most of that cost is the vehicle, not the data volume.** On identical
   evaluation windows and at equal training volume (control A), changing vehicle
   costs 0.13 PR-AUC in FS-B (0.975 to 0.864) and 0.36 in FS-A. Growing the
   cross-vehicle pool to its full size buys back 0.03 in FS-B and nothing in
   FS-A.
3. **Fleet diversity matters at fixed volume.** With the training volume held
   constant (control B), XGBoost in FS-B rises from 0.775 with one training
   vehicle to 0.875 with seven; in FS-A from 0.45 to 0.60.
4. **The representation decides the outcome.** Paired over 128 detector-vehicle
   cells, FS-B beats FS-A in 97% of them by a mean of 0.29 PR-AUC (p < 1e-4).
   The mechanism is a covariate shift so severe under FS-A that the feature
   space is not shared at all: on an unseen vehicle 99% of the identifier
   histogram falls in the overflow bucket. Aligning histogram columns by
   frequency rank instead (FS-A-rank) transfers better (0.641 vs 0.593 for
   XGBoost), which is a trap for anyone who does it by accident.
5. **Rules are a competitive floor and cover what learning misses.** The best
   rule is within 0.07 PR-AUC of the best learner. A one-line staleness rule
   catches 59% of suspension windows (67% of suspension events) cross-vehicle
   at 10 false alarms/hour; every learned detector scores at most 0.001 on that
   family, because the training pool holds a few hundred suspension windows
   among 75,000.
6. **Per-family recall is the operationally relevant view.** At 10 false
   alarms/hour the best learner catches 100% of flooding, 82% of timing
   manipulation, 83% of replay, 73% of fuzzing, 65% of spoofing and 60% of
   diagnostic windows. Masquerade is 0% for every frame-level detector by
   construction; ROAD, half of whose attacks are masquerade, is the hardest
   vehicle (0.33 for the trees, 0.60 for the Transformer, which is the one
   vehicle where the ranking inverts).
7. **Rankings are unstable and intervals overlap.** One detector moves 30 rank
   places depending only on the held-out vehicle; the 95% bootstrap intervals of
   every leading learned detector overlap. Eight vehicles support ranking
   representations and detector classes, not gradient-boosting libraries.

### 5.3 Figures

| File | Content |
|---|---|
| `results/figures/fig1_generalisation_gap.png` | In-domain vs cross-vehicle PR-AUC for every detector, both feature spaces |
| `results/figures/fig2_domain_heatmap.png` | Cross-vehicle PR-AUC by held-out vehicle |
| `results/figures/fig3_family_recall.png` | Recall by attack family at 10 FA/h |
| `results/figures/fig4_control_volume_matched.png` | Control A |
| `results/figures/fig5_control_fleet_size.png` | Control B |
| `results/figures/fig6_control_feature_ablation.png` | Control C |
| `results/figures/fig7_ci_forest.png` | Bootstrap intervals over vehicles |

## 6. Contributions and value to autonomous-vehicle cybersecurity

- **A cross-vehicle benchmark and protocol.** Eight real vehicle domains from
  three corpora under one schema and taxonomy, with a leave-one-vehicle-out
  protocol whose thresholds are set on benign traffic at a resolvable
  false-alarm budget. This turns an incomparable literature into a comparable
  one and gives a fixed yardstick for future detectors.
- **The first decomposition of the transfer penalty.** Volume-matched and
  fleet-size controls separate the vehicle change from training volume and
  fleet diversity, which no prior CAN study has done. The practical result for
  manufacturers is concrete: pool the fleet's data, in a representation where
  pooling is meaningful, and expect to recover most but not all of the gap.
- **A deployable representation result.** An identity-free encoding paired
  with a benign-traffic profile transfers to unseen vehicles and needs no
  labelled attack from the target. Benign driving data is free and abundant;
  physically verified attack data is scarce. A new model year can be covered
  from ordinary driving alone.
- **Suspension made detectable at frame level.** Two staleness statistics and a
  one-line rule detect message-deletion attacks on unseen vehicles, a family
  that arrival-based features cannot represent and that learned detectors miss.
- **An honest benchmark floor.** Deterministic rules in deployed scope and an
  unsupervised baseline let a reader see whether a learned model earned its
  complexity; the answer here is "partly, and not on every family", which
  argues for hybrid deployments.
- **Evaluation discipline the field can adopt.** Resolvability-checked
  false-alarm budgets, per-family and per-event recall, temporal splits that
  keep every family on both sides, stratified training sets with logged class
  counts, identical evaluation windows across representations, deterministic
  seeds, and bootstrap intervals on every headline number.
- **Reproducibility.** Resumable, parallel harness; every per-job metric row
  and log released; per-job score archives from which every metric can be
  recomputed without refitting.

## 7. Setup

### 7.1 Requirements

Python 3.11 or later (developed on 3.14) with `numpy`, `pandas`, `pyarrow`,
`scipy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `torch` (CPU is
sufficient), `matplotlib`, `python-docx`.

```bash
pip install numpy pandas pyarrow scipy scikit-learn xgboost lightgbm catboost torch matplotlib python-docx
```

Hardware used: 16-core laptop CPU, 64 GB RAM, no GPU. Peak memory with four
workers is about 12 GB. Put the working directory on local disk, not on a
synced or network drive.

### 7.2 Data

The Parquet corpus is included in `data_cache/` (1.6 GB, 217 captures plus
`manifest.json` and `cidv2_label_validation.csv`). Nothing needs downloading to
reproduce the benchmark.

To rebuild the corpus from the raw datasets, download:

- ROAD: https://0xsam.com/road/ (Zenodo record 10462796), place under `<RAW>/road/road/` with `ambient/` and `attacks/`
- CIDv2: https://data.4tu.nl/articles/dataset/Automotive_Controller_Area_Network_CAN_Bus_Intrusion_Dataset/12696950/2, place under `<RAW>/cidv2/<Vehicle>/*.log`
- can-train-and-test: https://bitbucket.org/brooke-lampe/can-train-and-test/src/master/, place under `<RAW>/ctt/set_0X/`

```bash
python src/build_cache.py     --raw <RAW> --out data_cache --only ROAD CIDv2
python src/build_ctt_cache.py --raw <RAW>/ctt --out data_cache --max-rows 400000
```

## 8. Reproducing the results

```bash
# 1. main benchmark: featurise once, then 48 jobs (8 vehicles x 2 feature spaces x {cross, in-domain fold 1, fold 2})
python src/run_benchmark.py --cache data_cache --work <LOCAL_WORK_DIR> --out results --workers 4 --threads 4

# 2. controls A, B, C (fast model subset; can run alongside step 1 once featurisation has finished)
python src/run_controls.py --work <LOCAL_WORK_DIR> --out results --workers 4 --threads 2

# 3. recompute only the deterministic rules for every job (seconds; used to add or change a rule)
python src/rerun_rules.py --work <LOCAL_WORK_DIR> --out results

# 4. tables, figures, bootstrap intervals, paired tests
python src/analyse.py --results results --work <LOCAL_WORK_DIR> --n-boot 200


```

Wall-clock on the hardware above: 5.2 hours for step 1, 5.0 hours for step 2
run concurrently, minutes for the rest. Both runs are resumable: a job whose
rows CSV already exists is skipped, so an interrupted run continues where it
stopped. All seeds derive from a CRC of the job name, so every split and
subsample is reproducible.

`<LOCAL_WORK_DIR>` receives the featurised windows (about 1.3 million windows,
memory-mapped by the workers) and the per-job score archives (about 1 GB), from
which every metric can be recomputed without refitting. The archives are not
committed here; `results/rows/` holds every per-job metric row.

## 9. Repository layout

```
README_v2.md                 this file
data_cache/                  Parquet corpus, manifest, CIDv2 label validation
src/
  canio.py                   loaders and label reconstruction for ROAD, CIDv2, can-train-and-test
  build_cache.py             ROAD and CIDv2 -> Parquet
  build_ctt_cache.py         streaming can-train-and-test -> Parquet, attack-centred slicing
  features.py                BenignProfile, FS-A (strict), FS-B (40 features)
  protocol.py                temporal quartering, folds, stratified subsampling, sequence indices
  baselines.py               rules (incl. staleness, robust-z and quantile unions), isolation forest
  models.py                  learned detectors and ensembles with real sequence context
  evaluate.py                PR-AUC, budgeted recall with resolvability flags, event recall, bootstrap
  bench_lib.py               featurisation store and per-job fit / score / evaluate loop
  run_benchmark.py           main protocol runner (resumable, parallel)
  run_controls.py            controls A (volume-matched), B (fleet size), C (feature ablation)
  rerun_rules.py             recompute rule detectors without refitting learned models
  analyse.py                 tables, figures, intervals, paired tests

results/
  results_raw.csv            every (protocol, fold, vehicle, feature space, detector) row: 992 rows
  table_headline.csv         in-domain vs cross-vehicle, per detector and feature space
  table_per_domain*.csv      PR-AUC by held-out vehicle
  table_per_family*.csv      recall, event recall and fixed-FPR recall by attack family
  table_control_A*.csv, table_control_B.csv, table_control_C.csv, controls_*.csv
  table_paired_tests.csv     Wilcoxon tests for every claimed comparison
  table_ci_domains.csv, table_ci_captures.csv   bootstrap intervals
  table_resolvability.csv    which false-alarm budgets each vehicle can resolve
  table_road_masquerade.csv  ROAD with and without masquerade windows
  table_rank_instability.csv
  figures/                   fig1 to fig7
  rows/                      per-job metric rows 
paper/                       the three documents listed at the top
```

## 10. Limitations

- Eight vehicle domains, one of them a bench rig; four share a dataset and
  therefore a collection methodology.
- CIDv2 attacks are post-hoc edits of a benign capture and several are ten
  frames long, so their per-vehicle estimates are high-variance.
- Masquerade attacks preserve every frame-level statistic by construction and
  are undetectable in this setting; they need the decoded-signal domain.


## 11. Citing and licensing

The three corpora carry their own licences and citation requirements (ROAD:
Verma et al., arXiv:2012.14600; CIDv2: Dupont et al., 4TU.ResearchData 2019;
can-train-and-test: Lampe and Meng, Computers & Security 2024). Please cite them
alongside this benchmark. Attack traffic in these corpora was produced on
dynamometers, benches or closed courses; nothing here should be replayed
against a vehicle on a public road or any system you do not own or have written
authorisation to test.
