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
| **RQ3** | Structural Property Identification | [`RQ3/`](RQ3/) | Can the model correctly identify graph-theoretic structural properties (hub regions, functional-network communities) directly from the raw edge list, and does handing it the correct structure (ground truth) help classification? |
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

| Scenario | Notebook | Prompt content |
|---|---|---|
| Scenario 1 — Base/Normal | `scenario1_base_normal.ipynb` | Vanilla hub prompt: ROI, degree, cohort z-score only. No prompt engineering, no network attribute. |
| Scenario 2 — Prompt (engineered) | `scenario2_prompt.ipynb` | Same fields as Scenario 1, plus an injected ADHD domain-knowledge sentence (reduced frontal-striatal hub strength, altered DMN hub connectivity). |
| Scenario 3 — Prompt + Net | `scenario3_prompt_net.ipynb` | Scenario 2's domain-knowledge sentence, plus each hub's Yeo-7 functional-network membership. |

**Scenario 1 (Base/Normal) prompt:**

```
Template: {DATASET_NAME} dataset, {PREPROCESS_TEMPLATE} atlas, 116 ROIs.
Description: The following are the top 20 positive ROI-degree-based hub regions identified from the subject's resting-state functional connectivity graph.
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
