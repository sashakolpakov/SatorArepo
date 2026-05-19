Overview
========

Problem Statement
-----------------

Given a text segment :math:`t`, the engine estimates whether the
segment carries evidence that is more compatible with human-written or
LLM-generated text in the engine's observable space.

The engine only sees statistical properties of transformer embedding
values. If human and LLM text overlap in those observables, the result
should be weak or ambiguous evidence, not a forced verdict.

This is a central product constraint, not a cosmetic label. Arepo must be
able to say "this score is weak" or "this window is ambiguous" when the
observable signal does not justify a hard authorship claim. This is what
prevents classic false positives such as treating the Declaration of
Independence as AI-generated merely because a single averaged score leans
AI-like.

Pipeline
--------

The production path is:

1. Split text into passages and overlapping word windows.
2. For each text unit, run four transformer models:

   * two generative models;
   * two masked/discriminative models.

3. Extract embedding values from each model.
4. Map embedding values to base-10 mantissas.
5. Compute one Kolmogorov-Smirnov statistic per model against
   :math:`Uniform(0,1)`.
6. Form a 4D feature vector.
7. Score the vector with an oriented Gaussian mixture of experts.
8. Compute geometric confidence from expert context-plane alignment.
9. Convert posterior direction plus geometric confidence into public
   evidence classes.
10. Aggregate windows into document-level evidence.

The production default model is stored at:

.. code-block:: text

   src/arepo/models/long_guardrail_4d_baseline_w120.npz

Output Philosophy
-----------------

The engine distinguishes three layers:

Posterior direction
   A numeric lean, such as :math:`p_{AI} = 0.55`.

Geometric confidence
   A reliability proxy based on whether the experts supporting the
   winning side are aligned with their own context geometry.

Evidence class
   The public interpretation: ``hard_ai``, ``hard_human``, ``soft_ai``,
   ``soft_human``, or ``ambiguous``.

A document can have :math:`p_{AI} = 0.55` while all local windows are
ambiguous. In that case the document evidence label should be
"Ambiguous evidence", not "Likely AI".

The confidence output is therefore part of the result, not an optional
decoration. Public consumers should read posterior scores together with
geometric confidence, hard-evidence coverage, and ambiguity coverage.
When these disagree, the evidence label is authoritative for the user
interface.

What The Engine Does Not Claim
------------------------------

The engine does not claim to detect authorship in every possible text.
Short boilerplate, formal email, encyclopedia prose, and template-like
answers can be genuinely non-identifiable under the current measurement.

Conversely, the engine should not hide behind ambiguity when hard local
evidence is present. The goal is calibrated evidence: high confidence
where the geometry is reliable, low confidence where human and AI text
overlap in the measured space.

The engine also does not expose internal expert names or votes through
the public UI. Public results are evidence summaries, probabilities,
geometric confidence values, and local window rankings.
