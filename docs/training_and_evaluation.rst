Training And Evaluation
=======================

Training Records
----------------

Training data is represented as records with:

.. code-block:: python

   {"text": "...", "label": 0 or 1, "group": "..."}

where ``label=0`` means human and ``label=1`` means AI. The group name
defines the source regime used to train source-local experts.

These are dataset labels, not proof of real authorship. Evaluation should
report source regimes and guardrail failures, not only global accuracy.

Expert Construction
-------------------

For each group, the trainer:

1. extracts 4D features;
2. learns orientation on a deterministic stratified validation split;
3. fits diagonal Gaussian human and AI class models;
4. fits pooled context geometry;
5. optionally fits local reliability tables;
6. saves the resulting experts into a ``.npz`` model file.

Optional expert expansion includes:

Dataset experts
   One expert per dataset/source group.

Cluster experts
   K-means subsets inside a source group, trained only when both classes
   have enough samples.

Dataset-union experts
   Experts trained on nearby dataset pairs in 4D context space.

Cluster-union experts
   Experts trained on nearby local clusters in 4D context space.

Scale-tagged experts
   Experts trained on fixed-size training windows, with group names such
   as ``wiki@w80`` or ``hc3@w120``. These exist because changing window
   length changes the empirical distribution and therefore the variance
   regime of the KS statistic.

The current production server does not train at request time. It loads
the saved MoE model.

Window-Scale Experiments
------------------------

The experiment branch ``math-evidence-windowing`` tested held-out
segmentation at whole-row, 80-word, 120-word, 160-word, and 240-word
scales. The conclusion was that scale mismatch matters: experts trained
on whole examples are not automatically competent on short local windows.

Operationally, this led to scale-tagged training-window support:

.. code-block:: text

   --train-window-words 80,120,240
   --train-window-stride-fraction 0.5
   --train-min-window-words 45

Accept scale-aware experts only when they improve held-out guardrails.

Guardrails
----------

Guardrails are test cases used to detect undesirable behavior, especially:

* public-domain human false positives;
* generated-text false negatives;
* short or formal text being forced into overconfident labels;
* whole-document averages contradicting local windows.

The built-in browser demo is not the full guardrail suite. It is a small
interactive smoke test.

Metrics
-------

The evaluation code reports:

* accuracy;
* precision;
* recall;
* F1;
* confusion counts;
* AUC-ROC from raw probability scores;
* hard-evidence coverage;
* accepted false-positive and false-negative behavior for gated reports.

Accuracy alone is not sufficient. Reports should also answer:

* How accurate are non-ambiguous hard-evidence calls?
* What coverage do those hard calls achieve?
* Which failures are false positives on protected human text?
* Which failures are false negatives on generated controls?
* Are failures clustered by source regime or window scale?

Useful Commands
---------------

.. code-block:: bash

   arepo-mixture
   arepo-guardrails
   arepo-evaluate
   arepo-report
   arepo-renorm
   arepo-thresholds

The CLI defaults should be checked before long runs; some experiments
load transformer models and can be slow.
