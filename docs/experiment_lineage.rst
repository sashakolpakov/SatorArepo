Experiment Lineage
==================

Why This Page Exists
--------------------

The ``math-evidence-windowing`` branch is an experiment branch. Its source
code is superseded by the current MoE-only branch, but its reports explain
several current design choices.

The branch should not be treated as the preferred implementation. It is a
research archive.

Useful Findings To Preserve
---------------------------

Mathematical underpinnings
   The branch includes a mathematical audit favoring local evidence over
   comparable windows instead of one forced document label. It also
   records why averaging posterior probabilities is not the same as
   aggregating evidence.

Geometric confidence
   The geometric-confidence reports separate posterior direction from
   confidence. High-confidence regions are more reliable but have lower
   coverage.

Window scale
   Window-length sweeps showed that 80-word windows were noisier, while
   120/160/240-word windows were closer. The conclusion was not to pick one
   universal window length, but to train and evaluate scale-aware experts
   such as ``wiki@w80`` and ``wiki@w120``.

Embedding vs 4D context
   Embedding context can fix additional failures, but also introduces new
   breaks and is much more expensive. The preserved conclusion is that 4D
   feature context should remain the default, while embedding context is a
   possible second-stage tool for residual failures.

Set coverage and residual experts
   The reports explored expert selection as a coverage problem: accept an
   expert when it fixes a residual slice without breaking protected human
   rows. This still requires held-out guardrails and caps on regressions.

What Was Superseded
-------------------

The branch predates the current MoE-only public server. Compared with the
current sprint branch, it lacks or is behind on:

* the explicit public evidence module;
* document evidence labels derived from local windows;
* the split hard/ambiguous LLM browser controls;
* the bundled production MoE model;
* the old-classifier API purge;
* the renamed MoE evaluator and training-data module;
* the Sphinx documentation structure.

Deletion Recommendation
-----------------------

Do not delete the branch until the useful reports have been distilled into
docs or intentionally archived. After that, it is safe to delete as a code
branch.

Before deletion, preserve at least these report topics in durable docs:

* mathematical underpinnings;
* geometric confidence threshold sweeps;
* all-values geometric confidence table;
* window-length findings;
* embedding-vs-feature context comparison;
* set-coverage and residual-expert conclusions.

The generated ``reports/`` directory should still remain out of normal
commits unless a specific report is intentionally cited as reproducibility
evidence.
