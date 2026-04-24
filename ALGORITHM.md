# Algorithm

This document describes the matching algorithm `glad-match` implements: the
problem it solves, the pipeline, the mathematical formulation of each stage,
the design choices behind those formulations, and the resulting privacy
properties.

## 1. Problem statement

Given:

- A **query** cohort $Q$ of $n_Q$ samples summarized as
  - Per-site alt/total allele counts on a fixed list of LD-pruned sites $\mathcal{S}_p$ (so the user shares aggregates only, never individual genotypes).
  - A Gaussian mixture model $G_Q$ fit in a feature space of the top $k$ principal components, optionally extended with a z-scored age dimension. When sex metadata is available, two GMMs $G_Q^{F}, G_Q^{M}$ are fit on the per-sex sub-cohorts.
  - Per-sex sample counts $(n_Q^F, n_Q^M)$ when sex is available.
- A **database** $D$ of $N$ samples with
  - PCA projections $\mathbf{x}_i \in \mathbb{R}^k$ (computed by the database operator on a fixed reference panel),
  - Age $a_i$ and sex $s_i \in \{F, M\}$,
  - Allele dosages $g_{i,s} \in \{0, 1, 2\}$ on $\mathcal{S}_p$ (with possible missingness),
  - Population labels and other categorical metadata for filtering.
- A target control count $n_C$.

Produce a subset $\mathcal{C} \subseteq D$ of size $n_C$ such that

1. The joint distribution of $\{(\mathbf{x}_i, a_i, s_i) : i \in \mathcal{C}\}$ is close to that of the query (controlling confounding by ancestry, age, and sex), and
2. The per-site case-vs-control allele-frequency test produces a well-calibrated null — formally, the genomic control statistic $\lambda \to 1$ on $\mathcal{S}_p$.

The constraint is that **no individual-level genotype, phenotype, or PCA projection** crosses from the user to the server. Inputs are aggregates and a fitted distribution; outputs are aggregates.

## 2. Pipeline overview

The algorithm has three stages:

1. **Candidate selection** — unbalanced Sinkhorn optimal transport from db samples to the query GMM in feature space yields a per-sample relevance score; the top-scoring samples form a pool of size $\alpha \cdot n_C$ (default $\alpha = 4$).
2. **Refinement** — greedy swap inside the pool, biased by relevance, targeting $(\log \lambda)^2 \to 0$ on the pruned site set with incremental $\chi^2$ updates.
3. **Aggregate output** — gzipped TSV of per-site case/control allele counts plus a sidecar JSON with pipeline diagnostics and a privacy-respecting view of the selected set's demographics.

When sex is part of the metadata, stages 1–2 run with the database partitioned by sex, and refinement only swaps within-sex so the per-sex ratio matches the query exactly.

## 3. Stage 1 — Candidate selection via unbalanced Sinkhorn OT

### 3.1 Feature space

Each query GMM lives in $\mathbb{R}^d$ where
$$d = k_{\text{used}} + \mathbb{1}[\text{age in mode}],$$
with $k_{\text{used}}$ the number of leading PCs the query side decided to use. The db is projected into the same space using the leading $k_{\text{used}}$ columns of its PCA matrix, with age z-scored against the database-wide $(\hat{\mu}_{\text{age}}, \hat{\sigma}_{\text{age}})$ stored in the manifest. (The z-score is the same one glad-prep applied to query ages, so both sides live in the same coordinate system.)

### 3.2 Cost matrix

For db sample $i$ and GMM component $k$ with mean $\boldsymbol{\mu}_k$ and full covariance $\boldsymbol{\Sigma}_k$,
$$C_{i,k} = (\mathbf{f}_i - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{f}_i - \boldsymbol{\Sigma}_k).$$

We never form $\boldsymbol{\Sigma}_k^{-1}$. Instead, for each component we Cholesky-decompose $\boldsymbol{\Sigma}_k = L_k L_k^\top$, then for each sample solve $L_k \mathbf{z} = \mathbf{f}_i - \boldsymbol{\mu}_k$ by forward substitution and set $C_{i,k} = \|\mathbf{z}\|_2^2$. This is numerically stable and cheap at our sizes ($d \leq \sim 31$). The implementation uses a hand-rolled Cholesky in pure Rust, avoiding a BLAS dependency.

The Mahalanobis cost is preferred over weighted Euclidean (with weights from variance-explained ratios) because the GMM's full covariance already encodes scale and correlation in feature space. PC weighting via variance explained would double-count; age vs. PC scaling is handled implicitly by the covariance.

### 3.3 Mass scaling and the unbalanced solver

We use unbalanced Sinkhorn with KL-penalized marginals (Chizat et al., 2018; Peyré & Cuturi, 2019). The objective minimized in the log-space scaling form is
$$\min_{P \geq 0} \; \langle P, C \rangle + \varepsilon\, \mathrm{KL}(P \,\|\, K) + \rho\, \mathrm{KL}(P\mathbf{1} \,\|\, \mathbf{a}) + \rho\, \mathrm{KL}(P^\top \mathbf{1} \,\|\, \mathbf{b}),$$
where $K_{i,k} = \exp(-C_{i,k}/\varepsilon)$. We pick:

- $\mathbf{a} = \tfrac{1}{N}\mathbf{1}$ — uniform probability over db samples.
- $\mathbf{b} = (w_1, \dots, w_K)$ — GMM mixture weights (already a probability vector).

Both marginals sum to 1. This is a *deliberate choice* of the probability scale at construction; the underlying solver explicitly does **not** normalize its inputs (the comment in `wass`'s `unbalanced_sinkhorn_log_with_convergence` warns that total mass is part of the signal in unbalanced OT). What that warning protects against is silently rescaling pre-existing meaningful masses; here we set the scale once and tune $(\varepsilon, \rho)$ against it.

The unbalanced formulation is essential: outlier db samples that fit no GMM component well have their row-mass damped by the $\rho \cdot \mathrm{KL}(P\mathbf{1} \| \mathbf{a})$ term, instead of being forced to ferry mass to the nearest component. This is what makes the candidate ranking robust to db samples that lie far outside the query's distribution.

Defaults: $\varepsilon = \mathrm{median}(C) / 50$ (auto), $\rho = 0.1$. The auto-$\varepsilon$ keeps the regularization cost-scale-relative; $\rho$ controls how much marginal mass the solver may discard in either direction. Both are CLI-overridable.

### 3.4 Relevance score and pool construction

Once the transport plan $P \in \mathbb{R}^{N \times K}$ converges, the per-sample relevance score is the row sum
$$r_i = \sum_{k=1}^{K} P_{i,k}.$$
Intuitively this is the total mass the GMM "sent" to db sample $i$ — high $r_i$ means the sample sits in a region of high query density; low $r_i$ means the unbalanced solver chose to discard mass rather than route it to $i$.

The candidate pool of size $\alpha \cdot n_C$ is the top-$r_i$ subset of the db. When the query is sex-split, we run two independent transport problems (one per sex) and allocate the pool proportionally:
$$\text{pool}_F = \mathrm{round}\!\left(\frac{n_Q^F}{n_Q^F + n_Q^M} \cdot \alpha \cdot n_C\right), \quad \text{pool}_M = \alpha n_C - \text{pool}_F.$$

## 4. Stage 2 — Refinement: greedy swap targeting $\lambda \to 1$

### 4.1 Test statistic and objective

For each site $s$ in the active set $\mathcal{S}_a = \mathcal{S}_p \cap \mathcal{S}_Q$ (the inner join of the pruned site list and what the query actually delivered counts for), the per-site $\chi^2$ on the 2×2 allele table is the standard Pearson statistic with 1 df:
$$\chi^2_s = \frac{N_s\,(a_s d_s - b_s c_s)^2}{(a_s + b_s)(c_s + d_s)(a_s + c_s)(b_s + d_s)},$$
where $(a_s, b_s) = (\mathrm{AC}_Q, \mathrm{AN}_Q - \mathrm{AC}_Q)$ and $(c_s, d_s) = (\mathrm{AC}_C, \mathrm{AN}_C - \mathrm{AC}_C)$ are the alt/ref counts in query and control, and $N_s = a_s + b_s + c_s + d_s$. Degenerate tables (zero margin or monomorphic) contribute 0.

The **genomic control** statistic (Devlin & Roeder, 1999) is
$$\lambda = \frac{\mathrm{median}_{s \in \mathcal{S}_a}(\chi^2_s)}{F^{-1}_{\chi^2_1}(0.5)} = \frac{\mathrm{median}(\chi^2)}{0.4549\ldots}.$$

The refinement objective is
$$\mathcal{L} = (\log \lambda)^2.$$

We target $\lambda = 1$, **not** $\min \lambda$. Minimizing $\lambda$ alone is the prior approach's failure mode: a deflated $\lambda < 1$ produces test statistics smaller than expected under the null, which is its own form of mis-calibration. The squared log penalty is symmetric around $\lambda = 1$ in log space and is the natural choice when the goal is calibration rather than power.

### 4.2 State and incremental updates

Refinement maintains
- $\mathcal{C} \subseteq \text{pool}$ — the current selected control set, $|\mathcal{C}| = n_C$.
- $\mathrm{AC}_C[s], \mathrm{AN}_C[s]$ for each $s \in \mathcal{S}_a$ — control allele counts.
- $\chi^2[s]$ for each $s$ — current per-site statistic.

The initial $\mathrm{AN}_C[s]$ is set to $2 |\mathcal{C}|$ assuming no missingness, and is decremented by 2 for each missing call encountered during the build pass. The initial $\mathrm{AC}_C[s]$ is the sum of dosages over selected samples at site $s$. Each $\chi^2[s]$ is computed from the resulting counts.

For a candidate swap (drop $d \in \mathcal{C}$, add $a \in \text{pool} \setminus \mathcal{C}$), only sites in the symmetric difference of their non-reference site sets change:
$$\mathcal{A}(d, a) = \mathrm{nonref}(d) \,\triangle\, \mathrm{nonref}(a) \;\cup\; \{s : \mathrm{nonref}(d) \cap \mathrm{nonref}(a) \ni s,\ g_{d,s} \neq g_{a,s}\}.$$
For each affected site we apply $\Delta\mathrm{AC}_C = g_{a,s} - g_{d,s}$ and (only if missingness changes) the corresponding $\Delta\mathrm{AN}_C$, then recompute $\chi^2_s$ from the four scalars. The genotype store is sample-major sparse, so iterating $\mathrm{nonref}(\cdot)$ is cheap and the size of $\mathcal{A}$ is at worst the number of non-ref sites carried by the two samples (typically $\sim 50\%$ of the active set).

Updating $\lambda$ means recomputing the median of a $\sim 30\mathrm{K}$-element vector; we use `select_nth_unstable` on an owned copy. This is the inner-loop hot path. A heap-or-tree-backed running median would amortize better but isn't yet warranted at our scale.

### 4.3 Greedy swap with transport-mass-weighted proposals

At each iteration we sample a small batch of swap proposals (default $b = 50$), evaluate each provisionally (compute affected-site changes and the resulting $\lambda$, *don't* mutate state), and apply only the single best improving swap. Mathematically: we accept the swap minimizing $\mathcal{L}$ over the batch, provided it strictly improves on the current $\mathcal{L}$.

Proposal sampling within a stratum biases toward where the OT relevance signal says the action should be:
$$\Pr(\text{drop } d) \propto (1 - \tilde r_d) + \phi, \qquad \Pr(\text{add } a) \propto \tilde r_a + \phi,$$
where $\tilde r$ is the relevance score normalized to $[0, 1]$ within the stratum and $\phi$ is a small uniform floor (default $0.05$) that prevents starvation of low-weight options. Drops favor "weakest fit" current controls; adds favor highest-relevance unselected candidates.

Termination is whichever of these comes first:

- $|\log \lambda| < \tau$ (default $\tau = 10^{-2}$, i.e. $\lambda$ within ~1% of 1),
- `plateau` consecutive batches with no improving swap (default 200),
- `max_iter` batches (default $10^4$).

When the query is sex-split, batches round-robin across strata and the drop / add are constrained to the same stratum. This preserves the per-sex selection ratio exactly through the entire refinement.

### 4.4 Why greedy and not simulated annealing

The prior approach used SA. SA is general but expensive: every proposal is evaluated and probabilistically accepted, and the cooling schedule is its own tuning surface. With incremental $\chi^2$ updates the per-evaluation cost is small enough that batched best-improvement greedy is both fast and well-defined. An SA temperature schedule would buy us the chance to escape local minima at the cost of throughput; with the OT-weighted proposal distribution biasing toward useful swaps, the local minima problem is muted in practice. A future variant could re-introduce a small amount of stochastic acceptance (a single annealing knob) if needed.

## 5. Stage 3 — Output

### 5.1 Per-site allele counts (TSV)

Columns: `chrom  pos  ref  alt  AC_case  AN_case  AC_ctrl  AN_ctrl`. One row per site in $\mathcal{S}_a$; gzipped. The control counts come from the final state of the refinement; the case counts are pulled directly from the query's reported aggregates. Output is bounded to the LD-pruned set because that is what glad-prep currently emits — extending to all sites is a glad-prep change, not a glad-match change.

The TSV format trades VCF's structured-multi-sample container for something a downstream pipeline can read in two lines of code. Because we never have individual-level controls to begin with, VCF's per-sample columns aren't useful.

### 5.2 Sidecar diagnostics (JSON)

A JSON file records:

- The seed and version metadata (for reproducibility),
- Per-stratum Sinkhorn convergence (iterations, objective, $\varepsilon$ used),
- Refinement diagnostics ($\lambda_{\text{init}} \to \lambda_{\text{final}}$, total iterations, accepted swaps),
- A privacy-respecting view of the selected set's demographics: per-sex counts, per-population counts, and an age histogram with k-anonymity suppression (cells with count $< k_{\min}$ are zeroed, with the suppressed total reported separately so downstream consumers know mass is hidden).

This is what enables the chi-square test on the user side without revealing any individual-level information about the selected controls.

## 6. Privacy properties

The aggregate-only contract is preserved end-to-end:

- **From user to server**: per-site allele counts (aggregate), per-sex sample counts (two integers), and a fitted GMM (parameters, not samples). The GMM is the only summary of the query's joint distribution that crosses the boundary, and it is a low-dimensional smoothing of the underlying point cloud.
- **Server-internal**: db genotypes and individual-level metadata are read but never returned.
- **From server to user**: per-site case/control allele counts and aggregate demographic summaries (binned, k-anon suppressed). No row ever corresponds to a specific db sample.

This boundary admits a chi-square test for case-control GWAS on the user side; covariate-adjusted regression would require stratified counts, which is a deliberate next-step extension subject to the same k-anonymity discipline.

## 7. Comparison to the prior approach

| | Prior (NN + SA) | Current (OT + greedy) |
|---|---|---|
| Distance | Weighted Mahalanobis on db sample $\times$ each query sample | Mahalanobis db sample $\times$ GMM components |
| Information shared by user | Individual PCA projections | Aggregate counts + GMM parameters |
| Age / sex handling | Ignored | Built into the GMM feature space; sex-stratified pipeline; per-sex ratio preserved exactly |
| Outlier robustness | NN is sensitive | Unbalanced OT damps outlier mass |
| Refinement | Simulated annealing on $\lambda$ | Greedy best-improvement-in-batch on $(\log \lambda)^2$ |
| Refinement cost | Full re-evaluation per proposal | Incremental $\chi^2$ update over affected sites |
| Objective | $\min \lambda$ (over-corrects) | $\lambda \to 1$ (calibration target) |
| Output | VCF with imputed counts | TSV of aggregate counts + JSON diagnostics |

## 8. Hyperparameters and defaults

| | Default | Note |
|---|---:|---|
| `pool_factor` $\alpha$ | 4 | Pool = $\alpha \cdot n_C$ |
| Sinkhorn $\varepsilon$ | $\mathrm{median}(C)/50$ (auto) | CLI `--sinkhorn-eps 0.0` |
| Sinkhorn $\rho$ | 0.1 | Allows ~10% marginal mass deletion |
| Sinkhorn `max_iter` | 1000 | |
| Sinkhorn `tol` | $10^{-6}$ | Dual update stability |
| Refine `max_iter` | $10^4$ | Iteration cap |
| Refine `batch` | 50 | Proposals per iteration |
| Refine `tol` | 0.01 | $\| \log \lambda \|$ stop criterion |
| Refine `plateau` | 200 | Consecutive non-improving batches |
| Refine `uniform_floor` $\phi$ | 0.05 | Proposal-weight floor |
| `k_anon_min` | 5 | Histogram cell suppression threshold |

## 9. Computational complexity

For the typical workload ($N \approx 40$K db samples per stratum, $K \in [4, 8]$ GMM components, $d \approx 31$, $|\mathcal{S}_a| \approx 30$K, $n_C \in [100, 5000]$):

- **Cost matrix**: $O(N K d)$ per stratum. With $N = 40000$, $K = 8$, $d = 31$, that's $\approx 10$M scalar Cholesky-solve ops — sub-second.
- **Sinkhorn**: $O(N K \cdot \mathrm{iters})$ per stratum; iters typically in the low hundreds.
- **Refinement initial state**: one pass over all selected samples' non-ref records $\to$ $O(n_C \cdot \overline{|\mathrm{nonref}|})$.
- **Refinement per iteration**: $O(b \cdot |\mathcal{A}|)$ chi-square recomputations + $O(b \cdot |\mathcal{S}_a|)$ for the median (the dominant term).
- **Refinement total**: bounded by $\mathrm{max\_iter} \cdot b \cdot |\mathcal{S}_a|$. In practice convergence happens well before the iteration cap.

End-to-end target on a workstation is single-digit minutes for production-sized inputs.

## 10. Limitations and future work

- **Strict sex / age matching** is not yet exposed. The filter layer is the right home: pre-restrict the db to a tight age window and/or only one sex, then run the existing pipeline.
- **phs-cohort exclusion** is wired through `FilterSpec` but waits on a `phs_cohort` column being added to the preprocessed db_pack.
- **Differential privacy** on the aggregate demographics output is not yet applied; k-anonymity suppression is a first cut.
- **BIC-based GMM component selection** in glad-prep would pick smaller $K$ for small queries and larger $K$ for big ones; the current heuristic ($K = n_Q / 200$, capped at 8) is a placeholder.
- **Refinement median structure**: a heap-or-tree-backed running median would replace the per-evaluation $O(|\mathcal{S}_a|)$ scan with $O(\log |\mathcal{S}_a|)$ updates if profiling identifies the median as a bottleneck on production-sized inputs.

## 11. References

- Browning, S. R., et al. *Ancestry-specific recent effective population size in the Americas.* PLOS Genetics, 2018. Demographic model used by `glad-sim`.
- Cuturi, M. *Sinkhorn distances: Lightspeed computation of optimal transport.* NeurIPS, 2013.
- Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.-X. *Scaling algorithms for unbalanced optimal transport problems.* Mathematics of Computation, 2018.
- Peyré, G., Cuturi, M. *Computational Optimal Transport.* Foundations and Trends in Machine Learning, 2019.
- Devlin, B., Roeder, K. *Genomic control for association studies.* Biometrics, 1999. Origin of the $\lambda$ statistic.

## Implementation map

Concrete entry points in this repository:

- Feature construction: `src/features.rs`
- Mahalanobis cost: `src/cost.rs`
- Sinkhorn wrapper: `src/ot.rs` (calls `wass::unbalanced_sinkhorn_log_with_convergence`)
- Candidate selection: `src/candidates.rs`
- Refinement loop: `src/refine.rs`
- Per-site $\chi^2$ and $\lambda$: `src/stats.rs`
- Filter hook: `src/filter.rs`
- Top-level orchestration: `src/pipeline.rs` (`match_controls`)
- TSV writer: `src/io/output_tsv.rs`
- Summary JSON: `src/io/summary_json.rs` + `pipeline::build_summary`
- CLI: `src/main.rs`
- Convenience preprocessing (raw artifacts → `db_pack/`): `build_db_pack.py`
