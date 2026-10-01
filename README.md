# Surface Cues, Not Structure: Probing LLM Behavior on Brain Connectivity Graphs

This repository contains the data pipeline, prompt variants, and analysis notebooks for a study
probing whether large language models (LLMs) reason over the actual **structure** of brain
functional-connectivity (FC) graphs when classifying ADHD vs. Control, or whether they instead
lean on **surface cues** — region names, prompt phrasing, and injected domain knowledge — that
have nothing to do with the graph topology itself.

All experiments use the **ADHD-200** resting-state fMRI dataset, parcellated with the
**AAL-116** atlas (116 regions of interest). Subjects are classified as `Control` or `ADHD` from
one of several text serializations of their FC graph, fed to a panel of LLMs (Llama-3.1-8B-Instruct,
DeepSeek-R1-Distill-Qwen-14B, Qwen2.5-14B-Instruct, Qwen3-14B, GPT-Luna, DeepSeek-V4-Flash).

## Research Questions

| RQ | Name | Folder | Question |
|----|------|--------|----------|
| **RQ1** | Knowledge Source | [`RQ_1_2/`](RQ_1_2/) | Does giving the model anatomical ROI *names* (vs. bare numeric node indices) change classification behavior — i.e. is the model leaning on prior world knowledge about brain regions rather than the connectivity structure? |
| **RQ2** | Statistical Context | [`RQ_1_2/`](RQ_1_2/) | Does the *statistical framing* of edge weights (raw Pearson correlation vs. cohort z-score) change classification behavior, independent of node identification? |
| **RQ3** | Structural Property Identification | [`RQ3/`](RQ3/) | Can the model correctly identify graph-theoretic structural properties (hub regions, functional-network connectivities) directly from the FC edge list, and does handing it the correct structure (ground truth) help classification? |
| **RQ4** | Perturbation Design | [`RQ4/`](RQ4/) | How sensitive is the model's prediction to *swapped* hub information — clinically-informed vs. random perturbation — and does that sensitivity change with prompt engineering or added network-membership context? |
| **RQ5** | Reasoning Fidelity | [`RQ5/`](RQ5/) | When the model produces a free-text rationale for its prediction, is that rationale actually *faithful* to the numeric graph evidence it was given, or is it a plausible-sounding but disconnected post-hoc story? |

RQ1 and RQ2 share a single experimental grid (node identity × edge statistic) and are implemented
together in [`RQ_1_2/`](RQ_1_2/).

---

## Repository Layout

```
preprocessing/    FC matrix extraction, baseline classifiers, structural ground truth (hubs/communities)
RQ_1_2/            RQ1 (Knowledge Source) x RQ2 (Statistical Context) -- 6 prompt scenarios
RQ3/               Structural Property Identification -- hub/community identification & ground-truth feeding
RQ4/               Perturbation Design -- hub swap experiments across 3 prompt scenarios x 5 perturbation cases
RQ5/               Reasoning Fidelity -- rationale vs. evidence comparison across models
```

### `preprocessing/`

- `fc_extraction_classification.ipynb` — builds the dataset: loads per-site phenotypic labels,
  reads `.1D` ROI time series, computes a 116×116 Pearson FC matrix per subject (averaged across
  runs), and runs baseline (non-LLM) classifiers. Also emits `aal_roi_names.json`, the canonical
  AAL-116 region-name list used by every prompt builder in this repo.
- `hub_community_ground_truth.ipynb` — computes the graph-theoretic ground truth used throughout
  RQ3–RQ5: positive/negative degree hubs (top-20, with cohort z-score), and within/between
  Yeo-7 functional-network edge counts (community structure), from each subject's top-10%
  positive edges.
- `subject_summary.csv` — per-subject metadata (site, DX label, counts) produced by the pipeline.

---

## RQ1 + RQ2 — Knowledge Source & Statistical Context

A 3×2 grid of prompt variants over the **same** underlying FC graph, isolating two independent
factors:

- **Node identification** (RQ1 — Knowledge Source): are ROI names available to the model, and if
  so, are they only given as a *legend in the prompt text* (surface knowledge) or embedded
  directly *inside the edge list itself* (structural)?
- **Edge statistic** (RQ2 — Statistical Context): are edge weights raw Pearson correlations, or
  cohort z-scores (deviation from the sample mean)?

| Scenario | Notebook | Node identification | Edge weight |
|---|---|---|---|
| scn1 | `fc_scn1.ipynb` | Numeric index only (no names anywhere) | Raw correlation |
| scn2 | `fc_named_prompt_scn2.ipynb` | Numeric index in the edge list + ROI-name **legend in the prompt** | Raw correlation |
| scn3 | `fc_named_scn3.ipynb` | Anatomical name **embedded directly in the edge list** | Raw correlation |
| scn4 | `zscore_scn4.ipynb` | Numeric index only | Cohort **z-score** |
| scn5 | `zscore_named_prompt_scn5.ipynb` | Numeric index + ROI-name **legend in the prompt** | Cohort **z-score** |
| scn6 | `named_zscore_scn6.ipynb` | Anatomical name **embedded directly in the edge list** | Cohort **z-score** |

`table.ipynb` aggregates results across all six scenarios and models.

**Shared prompt skeleton** (only the `Description` line and node/edge encoding change between
scenarios):
1. Description: The following is a functional connectivity brain graph for one subject, extracted from resting-state fMRI. Nodes represent brain regions, identified only by their numeric index (no region names provided); edges represent the strength of functional connectivity between two regions.
2. Description: The following is a functional connectivity brain graph for one subject, extracted from resting-state fMRI. Nodes represent brain regions; edges represent the strength of functional connectivity between two regions. Region names for each node index: Node 1=Precentral_L, Node 2=Precentral_R,....
3. Description: The following is a functional connectivity brain graph for one subject, extracted from resting-state fMRI. Nodes represent brain regions, identified by their anatomical region name (AAL atlas); edges represent the strength of functional connectivity between two named regions.
4. Description: The following is a functional connectivity brain graph for one subject, extracted from resting-state fMRI. Nodes represent brain regions, identified only by their numeric index (no region names provided); edges represent the connectivity strength between two regions as a cohort z-score (not a raw correlation). No node-level features are included, only the ranked list of edges.
5. Description: The following is a functional connectivity brain graph for one subject, extracted from resting-state fMRI. Nodes represent brain regions; edges represent the connectivity strength between two regions as a cohort z-score (not a raw correlation). No node-level features are included, only the ranked list of edges. Region names for each node index: Node 1=Precentral_L, Node 2=Precentral_R, ...
6. Description: The following is a functional connectivity brain graph for one subject, extracted from resting-state fMRI. Nodes represent brain regions; edges represent the connectivity strength between two regions as a cohort z-score (not a raw correlation), identified by their anatomical region name (AAL atlas). No node-level features are included, only the ranked list of edges. 
```
Template: This data comes from the {DATASET_NAME} dataset, preprocessed using the {PREPROCESS_TEMPLATE}
brain atlas template (116 regions of interest).
Description: <varies by scenario -- see table above>
Task: interpreting functional connectivity patterns and reasoning about the most likely diagnostic group
Request: Analyze the BrainGraph node features and edge list below. Find the main connectivity
patterns and the most important features. Then predict whether the subject belongs to Control or
ADHD. Give a confidence score (0-1) for both classes. Your output must strictly follow this JSON
structure and nothing else:
{
  "analysis": "a brief description of the overall connectivity pattern observed in the data (1-2 sentences)",
  "key_features": ["feature 1", "feature 2", "..."],
  "prediction": "Control or ADHD",
  "class_confidence": {"Control": 0.0, "ADHD": 0.0}
}
BrainGraph (Text format):
<edge list>
```

---

## RQ3 — Structural Property Identification

Two complementary prompt families, testing structural competence directly:

1. **Ground-truth feeding** (`hub_gt.ipynb`, `community_gt.ipynb`) — the model is *not* asked to
   compute anything. It is handed the precomputed structural ground truth (top-20 hub regions
   with degree + cohort z-score, or within/between Yeo-7 network connectivity counts) and asked
   to classify Control vs. ADHD directly from that structural summary.
2. **Self-identification** (`hub_identification.ipynb`, `community_identification.ipynb`) — the
   model is given the *raw* edge list and must first identify the structure itself, using a
   GraphArena-style (Tang et al., ICLR 2025) one-shot chain-of-thought prompt: `[task definition]
   -> [worked example on a toy graph] -> [problem to solve] -> [answer format]`. Its own
   self-identified hubs/communities are then optionally fed into an *independent* classification
   prompt (scored separately, so hub-finding accuracy doesn't contaminate classification scoring).

`comparison_table.ipynb` compares self-identified structure against the ground truth from
`preprocessing/hub_community_ground_truth.ipynb`.

**Hub-identification prompt (excerpt):**

```
You are required to identify hub regions in the given brain functional connectivity network and
output the top hub regions ranked by connectivity.

A hub region is a node with a high number of connections to other regions in the network. To
identify hubs, count how many times each region name appears across the connection list --
regions appearing most frequently are the most connected.

**Example**
- Regions in the network: Frontal_Sup_L, Frontal_Sup_R, Parietal_Inf_L, Occipital_Mid_L, Temporal_Sup_R
- Functional connections: Frontal_Sup_L to Parietal_Inf_L, Frontal_Sup_L to Occipital_Mid_L, Frontal_Sup_L to Temporal_Sup_R, Parietal_Inf_L to Occipital_Mid_L

In this network, we have the following regions and connections: Frontal_Sup_L, Frontal_Sup_R, Parietal_Inf_L, Occipital_Mid_L, and Temporal_Sup_R. The connections are: Frontal_Sup_L to Parietal_Inf_L, Frontal_Sup_L to Occipital_Mid_L, Frontal_Sup_L to Temporal_Sup_R, and Parietal_Inf_L to Occipital_Mid_L. Counting connections per region: Frontal_Sup_L appears 3 times, Parietal_Inf_L appears 2 times, Occipital_Mid_L appears 2 times, Temporal_Sup_R appears 1 time, and Frontal_Sup_R appears 0 times. The region with the highest connection count is Frontal_Sup_L. Therefore, the top hub is [Frontal_Sup_L].


**Problem to Solve**
- Regions in the network: Precentral_L, Precentral_R, Frontal_Sup_L, Frontal_Sup_R,....
- Functional connections: Temporal_Sup_L to Temporal_Sup_R, Rectus_L to Rectus_R, ...

Identify the top {k} hub regions in this network, ranked from most to least connected.
Present your answer in the following format: [Region1, Region2, ..., Region{k}]
```

**Classification prompt :**

```
COMMUNITY PROMPT (INPUT) ---
You are required to identify the most densely connected functional network pairs in the given brain functional connectivity network and output the top pairs ranked by connectivity.
Each region belongs to one of 8 functional groupings: the 7 Yeo networks (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default) plus 'Unassigned' for regions with no cortical Yeo-7 overlap. A network pair (same network twice for within-network, or two different networks for between-network) is highly connected if its regions appear frequently together across the connection list. To identify the top pairs, count how many times each network pair appears across the connection list -- pairs appearing most frequently are the most connected.

**Example**
- Regions and their networks: Frontal_Sup_L (Cont), Frontal_Sup_R (Cont), Parietal_Inf_L (DorsAttn), Occipital_Mid_L (Vis), Temporal_Sup_R (SalVentAttn)
- Functional connections: Frontal_Sup_L to Parietal_Inf_L, Frontal_Sup_R to Parietal_Inf_L, Frontal_Sup_L to Occipital_Mid_L, Frontal_Sup_L to Temporal_Sup_R, Parietal_Inf_L to Occipital_Mid_L, Frontal_Sup_L to Frontal_Sup_R
In this network, the regions belong to four functional networks: Cont (Frontal_Sup_L, Frontal_Sup_R), DorsAttn (Parietal_Inf_L), Vis (Occipital_Mid_L), and SalVentAttn (Temporal_Sup_R). Counting connections for each network pair (same network on both ends = within-network, different networks = between-network): Frontal_Sup_L to Parietal_Inf_L and Frontal_Sup_R to Parietal_Inf_L are both Cont-DorsAttn, so Cont-DorsAttn appears 2 times. Frontal_Sup_L to Occipital_Mid_L is Cont-Vis (1 time). Frontal_Sup_L to Temporal_Sup_R is Cont-SalVentAttn (1 time). Parietal_Inf_L to Occipital_Mid_L is DorsAttn-Vis (1 time). Frontal_Sup_L to Frontal_Sup_R is Cont-Cont, a within-network connection (1 time). Ranking all network pairs by connection count: the network pair with the highest number of connections is Cont-DorsAttn with 2 connections. Therefore, the top 5 network pairs are [Cont-DorsAttn, Cont-Vis, Cont-SalVentAttn, DorsAttn-Vis, Cont-Cont], where Cont-Cont is the only within-network pair among them.

**Problem to Solve**
- Regions and their networks: Precentral_L (SomMot), Precentral_R (SomMot), ...
- Functional connections: Temporal_Sup_L (SomMot) to Temporal_Sup_R (SomMot), Rectus_L (Limbic) to Rectus_R (Limbic), ...

Identify the top {top_k} network pairs in this network, ranked from most to least connected. Present your answer in the following format: [Pair1, Pair2, Pair3, ..., Pair{top_k}]"""

```

---

## RQ4 — Perturbation Design

Tests robustness of the classifier to *swapped* hub information: for each subject, the top hub
regions are perturbed by replacing them with either **clinically-informed** substitutes (regions
plausibly linked to ADHD literature) or **random** substitutes, at two perturbation depths
(**K5**, **K10**), plus a **no-perturbation** control. Perturbation input files live in
[`RQ4/inputs/`](RQ4/inputs/) (`hub_clinical_perturbed_k{5,10}.json`, `hub_random_perturbed_k{5,10}.json`,
and `_wnetwork` variants that also carry Yeo-7 network membership).

This 5-case perturbation grid (Clinical K5, Clinical K10, Random K5, Random K10, No Perturb) is
run across all **6 models** and **3 prompt scenarios** — the scenario notebooks hold the prompt
template fixed while the perturbation case (input file) and model vary:

| Scenario | Notebook | Prompt |
|---|---|---|
| Scenario 1 — Base/Normal | `scenario1_base_normal.ipynb` |  The following are the top 20 positive ROI-degree-based hub regions identified from the subject's resting-state functional connectivity graph. |
| Scenario 2 — Prompt (engineered) | `scenario2_prompt.ipynb` | The top 20 positive ROI-degree-based hub regions extracted from the subject's resting-state functional connectivity graph. ADHD has been associated with reduced frontal-striatal hub strength and altered default mode network (DMN) hub connectivity. Each hub includes its ROI, degree (number of positive connections), and cohort_z (deviation from the cohort mean; positive = above average, negative = below average). |
| Scenario 3 — Prompt + Net | `scenario3_prompt_net.ipynb` | The top 20 positive ROI-degree-based hub regions extracted from the subject's resting-state functional connectivity graph. ADHD has been associated with reduced frontal-striatal hub strength and altered default mode network (DMN) hub connectivity. Each hub includes its ROI, degree (number of positive connections), cohort_z (deviation from the cohort mean; positive = above average, negative = below average), and network (Yeo-7 functional network membership). |

**Scenario 1 (Base/Normal) prompt:**

```
Template: {DATASET_NAME} dataset, {PREPROCESS_TEMPLATE} atlas, 116 ROIs.
Description: <see above table>
Task: Based ONLY on the hub regions below, analyze the brain's most important hub regions and
predict whether the subject belongs to Control or ADHD. Provide a confidence score (0-1) for both classes.
Request: Your output must strictly follow this JSON structure and contain nothing else:
{
  "prediction": "Control or ADHD",
  "class_confidence": {"Control": 0.0, "ADHD": 0.0},
  "reasoning": "(exactly 1-2 short sentences, no more)"
}
Hub regions: Cingulum_Post_R - degree:22 cohort_z:2.75; Parietal_Sup_L - degree:20 cohort_z:2.68;...
```

**Scenario 2 (Prompt-engineered)** adds to the `Description`: *"ADHD has been associated with
reduced frontal-striatal hub strength and altered default mode network (DMN) hub connectivity."*

**Scenario 3 (Prompt + Net)** additionally appends each hub's `network:<Yeo-7 label>` field and
extends the domain-knowledge sentence to explain what `network` represents.

`hub_perturbation_inputs.ipynb` builds the perturbed input files; `RQ4_mcnemar_all_models.csv`
holds pairwise McNemar test results (perturbed vs. unperturbed predictions) across all
model × scenario × case combinations; `results/` contains the summary heatmap
(`heatmap_combined.png` / `.pdf`).




### Classification Accuracy Across Scenarios

Accuracy is calculated from each saved prediction file using scorable subjects only. Each percentage includes the correct/total count; ADHD and Control denominators include only scorable subjects from that ground-truth class. `No perturb` is the unmodified-hub control; K5/K10 rows use clinical or random hub perturbations.

<details>
<summary>DeepSeek-V4</summary>

| Scenario | Perturbation | Overall accuracy | ADHD accuracy | Control accuracy |
|:--|:--|--:|--:|--:|
| Hub | No perturb | 51.7% (394/762) | 44.4% (123/277) | 55.9% (271/485) |
| Hub | Clinical K5 | 47.3% (346/732) | 64.9% (172/265) | 37.3% (174/467) |
| Hub | Clinical K10 | 54.3% (401/738) | 61.4% (167/272) | 50.2% (234/466) |
| Hub | Random K5 | 48.6% (356/733) | 62.2% (166/267) | 40.8% (190/466) |
| Hub | Random K10 | 50.2% (368/733) | 61.5% (163/265) | 43.8% (205/468) |
| Hub + Prompt | No perturb | 56.2% (408/726) | 20.9% (55/263) | 76.2% (353/463) |
| Hub + Prompt | Clinical K5 | 57.8% (443/767) | 15.0% (42/280) | 82.3% (401/487) |
| Hub + Prompt | Clinical K10 | 59.2% (453/765) | 20.0% (56/280) | 81.9% (397/485) |
| Hub + Prompt | Random K5 | 56.2% (420/747) | 14.1% (38/269) | 79.9% (382/478) |
| Hub + Prompt | Random K10 | 59.7% (445/746) | 23.0% (62/270) | 80.5% (383/476) |
| Hub + Prompt + Net | No perturb | 54.1% (407/752) | 32.7% (90/275) | 66.5% (317/477) |
| Hub + Prompt + Net | Clinical K5 | 56.9% (436/766) | 31.2% (87/279) | 71.7% (349/487) |
| Hub + Prompt + Net | Clinical K10 | 57.0% (435/763) | 35.8% (100/279) | 69.2% (335/484) |
| Hub + Prompt + Net | Random K5 | 53.2% (407/765) | 28.8% (80/278) | 67.1% (327/487) |
| Hub + Prompt + Net | Random K10 | 55.9% (427/764) | 38.1% (106/278) | 66.0% (321/486) |

</details>

<details>
<summary>DeepSeek-R1-Distill</summary>

| Scenario | Perturbation | Overall accuracy | ADHD accuracy | Control accuracy |
|:--|:--|--:|--:|--:|
| Hub | No perturb | 50.5% (382/757) | 70.0% (194/277) | 39.2% (188/480) |
| Hub | Clinical K5 | 43.0% (321/747) | 83.3% (225/270) | 20.1% (96/477) |
| Hub | Clinical K10 | 39.9% (296/741) | 91.1% (245/269) | 10.8% (51/472) |
| Hub | Random K5 | 45.5% (285/626) | 51.5% (119/231) | 42.0% (166/395) |
| Hub | Random K10 | 42.5% (318/748) | 89.4% (245/274) | 15.4% (73/474) |
| Hub + Prompt | No perturb | 61.6% (395/641) | 18.3% (43/235) | 86.7% (352/406) |
| Hub + Prompt | Clinical K5 | 51.0% (293/575) | 53.7% (116/216) | 49.3% (177/359) |
| Hub + Prompt | Clinical K10 | 48.9% (266/544) | 66.8% (131/196) | 38.8% (135/348) |
| Hub + Prompt | Random K5 | 49.0% (302/616) | 52.2% (117/224) | 47.2% (185/392) |
| Hub + Prompt | Random K10 | 47.2% (250/530) | 63.6% (126/198) | 37.3% (124/332) |
| Hub + Prompt + Net | No perturb | 61.5% (441/717) | 2.7% (7/260) | 95.0% (434/457) |
| Hub + Prompt + Net | Clinical K5 | 53.5% (363/679) | 34.4% (87/253) | 64.8% (276/426) |
| Hub + Prompt + Net | Clinical K10 | 51.9% (337/649) | 54.2% (130/240) | 50.6% (207/409) |
| Hub + Prompt + Net | Random K5 | 52.5% (348/663) | 34.2% (80/234) | 62.5% (268/429) |
| Hub + Prompt + Net | Random K10 | 53.6% (327/610) | 41.9% (95/227) | 60.6% (232/383) |

</details>

<details>
<summary>Llama</summary>

| Scenario | Perturbation | Overall accuracy | ADHD accuracy | Control accuracy |
|:--|:--|--:|--:|--:|
| Hub | No perturb | 52.9% (406/768) | 41.4% (116/280) | 59.4% (290/488) |
| Hub | Clinical K5 | 45.8% (352/768) | 66.4% (186/280) | 34.0% (166/488) |
| Hub | Clinical K10 | 49.5% (380/768) | 58.9% (165/280) | 44.1% (215/488) |
| Hub | Random K5 | 50.3% (386/768) | 58.9% (165/280) | 45.3% (221/488) |
| Hub | Random K10 | 53.6% (412/768) | 45.0% (126/280) | 58.6% (286/488) |
| Hub + Prompt | No perturb | 49.5% (380/768) | 53.2% (149/280) | 47.3% (231/488) |
| Hub + Prompt | Clinical K5 | 43.9% (337/768) | 78.2% (219/280) | 24.2% (118/488) |
| Hub + Prompt | Clinical K10 | 44.7% (343/768) | 68.9% (193/280) | 30.7% (150/488) |
| Hub + Prompt | Random K5 | 42.2% (324/768) | 75.0% (210/280) | 23.4% (114/488) |
| Hub + Prompt | Random K10 | 47.1% (362/768) | 50.4% (141/280) | 45.3% (221/488) |
| Hub + Prompt + Net | No perturb | 56.0% (430/768) | 33.9% (95/280) | 68.6% (335/488) |
| Hub + Prompt + Net | Clinical K5 | 51.6% (396/768) | 38.2% (107/280) | 59.2% (289/488) |
| Hub + Prompt + Net | Clinical K10 | 54.0% (415/768) | 48.6% (136/280) | 57.2% (279/488) |
| Hub + Prompt + Net | Random K5 | 51.2% (393/768) | 33.9% (95/280) | 61.1% (298/488) |
| Hub + Prompt + Net | Random K10 | 55.3% (425/768) | 27.9% (78/280) | 71.1% (347/488) |

</details>

<details>
<summary>Qwen2.5</summary>

| Scenario | Perturbation | Overall accuracy | ADHD accuracy | Control accuracy |
|:--|:--|--:|--:|--:|
| Hub | No perturb | 63.5% (488/768) | 1.8% (5/280) | 99.0% (483/488) |
| Hub | Clinical K5 | 63.0% (484/768) | 1.1% (3/280) | 98.6% (481/488) |
| Hub | Clinical K10 | 63.0% (484/768) | 0.7% (2/280) | 98.8% (482/488) |
| Hub | Random K5 | 63.4% (487/768) | 2.5% (7/280) | 98.4% (480/488) |
| Hub | Random K10 | 63.5% (488/768) | 0.7% (2/280) | 99.6% (486/488) |
| Hub + Prompt | No perturb | 63.4% (487/768) | 0.0% (0/280) | 99.8% (487/488) |
| Hub + Prompt | Clinical K5 | 62.6% (481/768) | 2.9% (8/280) | 96.9% (473/488) |
| Hub + Prompt | Clinical K10 | 63.2% (485/768) | 2.9% (8/280) | 97.7% (477/488) |
| Hub + Prompt | Random K5 | 62.8% (482/768) | 3.6% (10/280) | 96.7% (472/488) |
| Hub + Prompt | Random K10 | 57.2% (439/768) | 8.9% (25/280) | 84.8% (414/488) |
| Hub + Prompt + Net | No perturb | 63.5% (488/768) | 0.0% (0/280) | 100.0% (488/488) |
| Hub + Prompt + Net | Clinical K5 | 63.7% (489/768) | 0.4% (1/280) | 100.0% (488/488) |
| Hub + Prompt + Net | Clinical K10 | 63.3% (486/768) | 1.1% (3/280) | 99.0% (483/488) |
| Hub + Prompt + Net | Random K5 | 63.8% (490/768) | 1.1% (3/280) | 99.8% (487/488) |
| Hub + Prompt + Net | Random K10 | 61.8% (475/768) | 5.0% (14/280) | 94.5% (461/488) |

</details>

<details>
<summary>Qwen3</summary>

| Scenario | Perturbation | Overall accuracy | ADHD accuracy | Control accuracy |
|:--|:--|--:|--:|--:|
| Hub | No perturb | 41.1% (316/768) | 87.1% (244/280) | 14.8% (72/488) |
| Hub | Clinical K5 | 41.1% (316/768) | 93.6% (262/280) | 11.1% (54/488) |
| Hub | Clinical K10 | 38.5% (296/768) | 92.9% (260/280) | 7.4% (36/488) |
| Hub | Random K5 | 50.9% (391/768) | 65.0% (182/280) | 42.8% (209/488) |
| Hub | Random K10 | 38.4% (295/768) | 88.6% (248/280) | 9.6% (47/488) |
| Hub + Prompt | No perturb | 58.1% (446/768) | 13.9% (39/280) | 83.4% (407/488) |
| Hub + Prompt | Clinical K5 | 46.7% (359/768) | 69.6% (195/280) | 33.6% (164/488) |
| Hub + Prompt | Clinical K10 | 42.8% (329/768) | 86.1% (241/280) | 18.0% (88/488) |
| Hub + Prompt | Random K5 | 47.0% (361/768) | 64.3% (180/280) | 37.1% (181/488) |
| Hub + Prompt | Random K10 | 41.7% (320/768) | 80.0% (224/280) | 19.7% (96/488) |
| Hub + Prompt + Net | No perturb | 57.8% (444/768) | 20.7% (58/280) | 79.1% (386/488) |
| Hub + Prompt + Net | Clinical K5 | 42.7% (328/768) | 77.5% (217/280) | 22.7% (111/488) |
| Hub + Prompt + Net | Clinical K10 | 39.3% (302/768) | 91.1% (255/280) | 9.6% (47/488) |
| Hub + Prompt + Net | Random K5 | 41.8% (321/768) | 70.7% (198/280) | 25.2% (123/488) |
| Hub + Prompt + Net | Random K10 | 41.9% (322/768) | 86.1% (241/280) | 16.6% (81/488) |

</details>

<details>
<summary>GPT-Luna</summary>

| Scenario | Perturbation | Overall accuracy | ADHD accuracy | Control accuracy |
|:--|:--|--:|--:|--:|
| Hub | No perturb | 41.3% (317/768) | 90.0% (252/280) | 13.3% (65/488) |
| Hub | Clinical K5 | 41.2% (316/767) | 83.6% (234/280) | 16.8% (82/487) |
| Hub | Clinical K10 | 40.7% (312/766) | 86.0% (239/278) | 15.0% (73/488) |
| Hub | Random K5 | 42.7% (326/763) | 82.7% (230/278) | 19.8% (96/485) |
| Hub | Random K10 | 37.5% (288/767) | 95.7% (267/279) | 4.3% (21/488) |
| Hub + Prompt | No perturb | 53.5% (411/768) | 48.6% (136/280) | 56.4% (275/488) |
| Hub + Prompt | Clinical K5 | 43.6% (334/766) | 80.0% (224/280) | 22.6% (110/486) |
| Hub + Prompt | Clinical K10 | 44.8% (344/768) | 87.5% (245/280) | 20.3% (99/488) |
| Hub + Prompt | Random K5 | 47.2% (362/767) | 63.4% (177/279) | 37.9% (185/488) |
| Hub + Prompt | Random K10 | 44.1% (338/766) | 66.8% (187/280) | 31.1% (151/486) |
| Hub + Prompt + Net | No perturb | 52.9% (406/768) | 48.9% (137/280) | 55.1% (269/488) |
| Hub + Prompt + Net | Clinical K5 | 41.0% (315/768) | 85.4% (239/280) | 15.6% (76/488) |
| Hub + Prompt + Net | Clinical K10 | 41.0% (315/768) | 93.6% (262/280) | 10.9% (53/488) |
| Hub + Prompt + Net | Random K5 | 46.5% (357/768) | 67.1% (188/280) | 34.6% (169/488) |
| Hub + Prompt + Net | Random K10 | 44.8% (343/765) | 74.6% (208/279) | 27.8% (135/486) |

</details>

### Flip Rates and McNemar Test Results

Results cover 768 subjects. Flip rate is the proportion of scorable subjects whose prediction changed. `b` and `c` are the discordant-pair counts used by McNemar's test. Bold p-values indicate `p < 0.05`; values shown as `<0.0001` were printed as `0.0000` in the source output.

<details>
<summary>DeepSeek-V4</summary>

| Format | K | Comparison | Flip rate | b | c | p-value | Scorable | Skipped |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Hub | 5 | Baseline vs Clinical | 46.8% | 187 | 153 | 0.0734 | 726 | 42 |
| Hub | 5 | Random vs Clinical | 45.8% | 167 | 153 | 0.4675 | 698 | 70 |
| Hub | 5 | Baseline vs Random | 45.5% | 176 | 155 | 0.2716 | 728 | 40 |
| Hub | 10 | Baseline vs Clinical | 47.7% | 164 | 186 | 0.2616 | 733 | 35 |
| Hub | 10 | Random vs Clinical | 46.8% | 152 | 178 | 0.1687 | 705 | 63 |
| Hub | 10 | Baseline vs Random | 47.0% | 178 | 164 | 0.4821 | 727 | 41 |
| Hub + Prompt | 5 | Baseline vs Clinical | 30.6% | 104 | 118 | 0.3830 | 725 | 43 |
| Hub + Prompt | 5 | Random vs Clinical | 25.6% | 88 | 103 | 0.3111 | 746 | 22 |
| Hub + Prompt | 5 | Baseline vs Random | 31.5% | 112 | 111 | 1.0000 | 707 | 61 |
| Hub + Prompt | 10 | Baseline vs Clinical | 32.9% | 106 | 132 | 0.1049 | 723 | 45 |
| Hub + Prompt | 10 | Random vs Clinical | 28.8% | 108 | 106 | 0.9455 | 743 | 25 |
| Hub + Prompt | 10 | Baseline vs Random | 32.0% | 101 | 124 | 0.1423 | 704 | 64 |
| Hub + Prompt + Net | 5 | Baseline vs Clinical | 42.7% | 149 | 171 | 0.2404 | 750 | 18 |
| Hub + Prompt + Net | 5 | Random vs Clinical | 42.3% | 147 | 176 | 0.1191 | 763 | 5 |
| Hub + Prompt + Net | 5 | Baseline vs Random | 43.1% | 164 | 159 | 0.8239 | 749 | 19 |
| Hub + Prompt + Net | 10 | Baseline vs Clinical | 45.6% | 158 | 183 | 0.1936 | 747 | 21 |
| Hub + Prompt + Net | 10 | Random vs Clinical | 45.1% | 167 | 175 | 0.7051 | 759 | 9 |
| Hub + Prompt + Net | 10 | Baseline vs Random | 45.5% | 164 | 176 | 0.5509 | 748 | 20 |

</details>

<details>
<summary>R1-Distill</summary>

| Format | K | Comparison | Flip rate | b | c | p-value | Scorable | Skipped |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Hub | 5 | Baseline vs Clinical | 40.5% | 175 | 123 | **0.0031** | 736 | 32 |
| Hub | 5 | Random vs Clinical | 46.1% | 147 | 133 | 0.4373 | 608 | 160 |
| Hub | 5 | Baseline vs Random | 49.3% | 170 | 135 | 0.0514 | 619 | 149 |
| Hub | 10 | Baseline vs Clinical | 38.9% | 181 | 103 | **<0.0001** | 731 | 37 |
| Hub | 10 | Random vs Clinical | 21.0% | 84 | 68 | 0.2236 | 723 | 45 |
| Hub | 10 | Baseline vs Random | 40.6% | 179 | 120 | **0.0008** | 737 | 31 |
| Hub + Prompt | 5 | Baseline vs Clinical | 49.5% | 139 | 95 | **0.0048** | 473 | 295 |
| Hub + Prompt | 5 | Random vs Clinical | 49.4% | 111 | 118 | 0.6918 | 464 | 304 |
| Hub + Prompt | 5 | Baseline vs Random | 51.1% | 160 | 107 | **0.0014** | 523 | 245 |
| Hub + Prompt | 10 | Baseline vs Clinical | 60.4% | 170 | 106 | **0.0001** | 457 | 311 |
| Hub + Prompt | 10 | Random vs Clinical | 43.7% | 75 | 88 | 0.3473 | 373 | 395 |
| Hub + Prompt | 10 | Baseline vs Random | 58.5% | 158 | 99 | **0.0003** | 439 | 329 |
| Hub + Prompt + Net | 5 | Baseline vs Clinical | 37.6% | 140 | 98 | **0.0077** | 633 | 135 |
| Hub + Prompt + Net | 5 | Random vs Clinical | 39.2% | 113 | 118 | 0.7925 | 589 | 179 |
| Hub + Prompt + Net | 5 | Baseline vs Random | 37.3% | 146 | 86 | **0.0001** | 622 | 146 |
| Hub + Prompt + Net | 10 | Baseline vs Clinical | 51.8% | 189 | 126 | **0.0005** | 608 | 160 |
| Hub + Prompt + Net | 10 | Random vs Clinical | 45.4% | 125 | 111 | 0.3975 | 520 | 248 |
| Hub + Prompt + Net | 10 | Baseline vs Random | 40.2% | 134 | 93 | **0.0078** | 564 | 204 |

</details>

<details>
<summary>Llama</summary>

| Format | K | Comparison | Flip rate | b | c | p-value | Scorable | Skipped |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Hub | 5 | Baseline vs Clinical | 44.3% | 197 | 143 | **0.0040** | 768 | 0 |
| Hub | 5 | Random vs Clinical | 41.4% | 176 | 142 | 0.0641 | 768 | 0 |
| Hub | 5 | Baseline vs Random | 33.3% | 138 | 118 | 0.2350 | 768 | 0 |
| Hub | 10 | Baseline vs Clinical | 48.7% | 200 | 174 | 0.1960 | 768 | 0 |
| Hub | 10 | Random vs Clinical | 45.3% | 190 | 158 | 0.0964 | 768 | 0 |
| Hub | 10 | Baseline vs Random | 45.3% | 171 | 177 | 0.7887 | 768 | 0 |
| Hub + Prompt | 5 | Baseline vs Clinical | 42.3% | 184 | 141 | **0.0197** | 768 | 0 |
| Hub + Prompt | 5 | Random vs Clinical | 31.4% | 114 | 127 | 0.4396 | 768 | 0 |
| Hub + Prompt | 5 | Baseline vs Random | 38.0% | 174 | 118 | **0.0012** | 768 | 0 |
| Hub + Prompt | 10 | Baseline vs Clinical | 47.5% | 201 | 164 | 0.0594 | 768 | 0 |
| Hub + Prompt | 10 | Random vs Clinical | 44.7% | 181 | 162 | 0.3311 | 768 | 0 |
| Hub + Prompt | 10 | Baseline vs Random | 44.3% | 179 | 161 | 0.3566 | 768 | 0 |
| Hub + Prompt + Net | 5 | Baseline vs Clinical | 38.3% | 164 | 130 | 0.0541 | 768 | 0 |
| Hub + Prompt + Net | 5 | Random vs Clinical | 40.2% | 153 | 156 | 0.9094 | 768 | 0 |
| Hub + Prompt + Net | 5 | Baseline vs Random | 38.9% | 168 | 131 | **0.0372** | 768 | 0 |
| Hub + Prompt + Net | 10 | Baseline vs Clinical | 43.1% | 173 | 158 | 0.4416 | 768 | 0 |
| Hub + Prompt + Net | 10 | Random vs Clinical | 44.5% | 176 | 166 | 0.6266 | 768 | 0 |
| Hub + Prompt + Net | 10 | Baseline vs Random | 40.8% | 159 | 154 | 0.8212 | 768 | 0 |

</details>

<details>
<summary>Qwen2.5</summary>

| Format | K | Comparison | Flip rate | b | c | p-value | Scorable | Skipped |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Hub | 5 | Baseline vs Clinical | 2.3% | 11 | 7 | 0.4807 | 768 | 0 |
| Hub | 5 | Random vs Clinical | 3.0% | 13 | 10 | 0.6776 | 768 | 0 |
| Hub | 5 | Baseline vs Random | 2.5% | 10 | 9 | 1.0000 | 768 | 0 |
| Hub | 10 | Baseline vs Clinical | 2.3% | 11 | 7 | 0.4807 | 768 | 0 |
| Hub | 10 | Random vs Clinical | 1.6% | 8 | 4 | 0.3877 | 768 | 0 |
| Hub | 10 | Baseline vs Random | 1.8% | 7 | 7 | 1.0000 | 768 | 0 |
| Hub + Prompt | 5 | Baseline vs Clinical | 3.1% | 15 | 9 | 0.3075 | 768 | 0 |
| Hub + Prompt | 5 | Random vs Clinical | 5.9% | 23 | 22 | 1.0000 | 768 | 0 |
| Hub + Prompt | 5 | Baseline vs Random | 3.3% | 15 | 10 | 0.4244 | 768 | 0 |
| Hub + Prompt | 10 | Baseline vs Clinical | 2.6% | 11 | 9 | 0.8238 | 768 | 0 |
| Hub + Prompt | 10 | Random vs Clinical | 13.8% | 30 | 76 | **<0.0001** | 768 | 0 |
| Hub + Prompt | 10 | Baseline vs Random | 13.0% | 74 | 26 | **<0.0001** | 768 | 0 |
| Hub + Prompt + Net | 5 | Baseline vs Clinical | 0.1% | 0 | 1 | 1.0000 | 768 | 0 |
| Hub + Prompt + Net | 5 | Random vs Clinical | 0.7% | 3 | 2 | 1.0000 | 768 | 0 |
| Hub + Prompt + Net | 5 | Baseline vs Random | 0.5% | 1 | 3 | 0.6250 | 768 | 0 |
| Hub + Prompt + Net | 10 | Baseline vs Clinical | 1.0% | 5 | 3 | 0.7266 | 768 | 0 |
| Hub + Prompt + Net | 10 | Random vs Clinical | 6.1% | 18 | 29 | 0.1439 | 768 | 0 |
| Hub + Prompt + Net | 10 | Baseline vs Random | 5.3% | 27 | 14 | 0.0596 | 768 | 0 |

</details>

<details>
<summary>Qwen3</summary>

| Format | K | Comparison | Flip rate | b | c | p-value | Scorable | Skipped |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Hub | 5 | Baseline vs Clinical | 21.4% | 82 | 82 | 1.0000 | 768 | 0 |
| Hub | 5 | Random vs Clinical | 40.2% | 192 | 117 | **<0.0001** | 768 | 0 |
| Hub | 5 | Baseline vs Random | 38.4% | 110 | 185 | **<0.0001** | 768 | 0 |
| Hub | 10 | Baseline vs Clinical | 18.5% | 81 | 61 | 0.1105 | 768 | 0 |
| Hub | 10 | Random vs Clinical | 16.0% | 61 | 62 | 1.0000 | 768 | 0 |
| Hub | 10 | Baseline vs Random | 20.4% | 89 | 68 | 0.1102 | 768 | 0 |
| Hub + Prompt | 5 | Baseline vs Clinical | 58.5% | 268 | 181 | **<0.0001** | 768 | 0 |
| Hub + Prompt | 5 | Random vs Clinical | 38.8% | 150 | 148 | 0.9538 | 768 | 0 |
| Hub + Prompt | 5 | Baseline vs Random | 54.8% | 253 | 168 | **<0.0001** | 768 | 0 |
| Hub + Prompt | 10 | Baseline vs Clinical | 72.5% | 337 | 220 | **<0.0001** | 768 | 0 |
| Hub + Prompt | 10 | Random vs Clinical | 25.9% | 95 | 104 | 0.5708 | 768 | 0 |
| Hub + Prompt | 10 | Baseline vs Random | 67.4% | 322 | 196 | **<0.0001** | 768 | 0 |
| Hub + Prompt + Net | 5 | Baseline vs Clinical | 62.8% | 299 | 183 | **<0.0001** | 768 | 0 |
| Hub + Prompt + Net | 5 | Random vs Clinical | 34.5% | 129 | 136 | 0.7125 | 768 | 0 |
| Hub + Prompt + Net | 5 | Baseline vs Random | 62.4% | 301 | 178 | **<0.0001** | 768 | 0 |
| Hub + Prompt + Net | 10 | Baseline vs Clinical | 74.5% | 357 | 215 | **<0.0001** | 768 | 0 |
| Hub + Prompt + Net | 10 | Random vs Clinical | 20.8% | 90 | 70 | 0.1328 | 768 | 0 |
| Hub + Prompt + Net | 10 | Baseline vs Random | 70.8% | 333 | 211 | **<0.0001** | 768 | 0 |

</details>

<details>
<summary>Luna</summary>

| Format | K | Comparison | Flip rate | b | c | p-value | Scorable | Skipped |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Hub | 5 | Baseline vs Clinical | 20.2% | 78 | 77 | 1.0000 | 767 | 1 |
| Hub | 5 | Random vs Clinical | 23.9% | 97 | 85 | 0.4149 | 762 | 6 |
| Hub | 5 | Baseline vs Random | 18.6% | 65 | 77 | 0.3560 | 763 | 5 |
| Hub | 10 | Baseline vs Clinical | 22.6% | 88 | 85 | 0.8792 | 766 | 2 |
| Hub | 10 | Random vs Clinical | 17.1% | 53 | 78 | **0.0356** | 765 | 3 |
| Hub | 10 | Baseline vs Random | 13.0% | 64 | 36 | **0.0066** | 767 | 1 |
| Hub + Prompt | 5 | Baseline vs Clinical | 36.7% | 179 | 102 | **<0.0001** | 766 | 2 |
| Hub + Prompt | 5 | Random vs Clinical | 28.4% | 123 | 94 | 0.0571 | 765 | 3 |
| Hub + Prompt | 5 | Baseline vs Random | 26.1% | 124 | 76 | **0.0008** | 767 | 1 |
| Hub + Prompt | 10 | Baseline vs Clinical | 43.6% | 201 | 134 | **0.0003** | 768 | 0 |
| Hub + Prompt | 10 | Random vs Clinical | 27.7% | 103 | 109 | 0.7314 | 766 | 2 |
| Hub + Prompt | 10 | Baseline vs Random | 32.5% | 160 | 89 | **<0.0001** | 766 | 2 |
| Hub + Prompt + Net | 5 | Baseline vs Clinical | 39.5% | 197 | 106 | **<0.0001** | 768 | 0 |
| Hub + Prompt + Net | 5 | Random vs Clinical | 24.7% | 116 | 74 | **0.0028** | 768 | 0 |
| Hub + Prompt + Net | 5 | Baseline vs Random | 24.9% | 120 | 71 | **0.0005** | 768 | 0 |
| Hub + Prompt + Net | 10 | Baseline vs Clinical | 46.2% | 223 | 132 | **<0.0001** | 768 | 0 |
| Hub + Prompt + Net | 10 | Random vs Clinical | 24.7% | 109 | 80 | **0.0414** | 765 | 3 |
| Hub + Prompt + Net | 10 | Baseline vs Random | 34.6% | 163 | 102 | **0.0002** | 765 | 3 |

</details>

---

## RQ5 — Reasoning Fidelity (`RQ5/`)

Samples 10 subjects (5 ADHD, 5 Control) from the RQ3 community ground-truth classification run
and checks, per model, whether the free-text `"reasoning"` the model produced is actually
consistent with the numeric within-/between-network connectivity counts it was given — or whether
it's a fluent but evidence-disconnected justification. Compared across three models: DeepSeek-V4,
GPT-Luna, and Llama.

- `generate_file.ipynb` — selects the matched 5 ADHD / 5 Control subject sample per model and
  merges each model's prediction + reasoning with the ground-truth community counts.
- `deepseek_v4_community_gt_5adhd_5control_10com.json`,
  `gpt_luna_community_gt_5adhd_5control_10com.json`,
  `llama_community_gt_5adhd_5control_10com.json` — the resulting per-model, per-subject records:
  `subject_id`, `ground_truth`, `prediction`, `within_network` counts, `between_network` counts,
  and the model's free-text `reasoning`, ready for manual/qualitative fidelity comparison against
  the numeric evidence.

---

## Output Conventions

Every inference notebook shares the same JSON-output contract, enforced via a system prompt
(*"Output only one valid JSON object matching the required schema. No markdown, comments, or
extra text. Must parse with `json.loads()`."*) and validated with a sanity check before the full
run: `prediction ∈ {Control, ADHD}`, a `class_confidence` dict with both classes, and a short
`reasoning` string. Runs checkpoint every `N` subjects (`*_checkpoint.json`) and write a final
merged results file (`*_final.json` / scenario-named equivalent) plus an error log.
