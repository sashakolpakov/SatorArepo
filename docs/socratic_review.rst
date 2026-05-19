Socratic Review Checklist
=========================

This page is intentionally written as a review prompt. Use it after
reading the docs and after inspecting a few live examples.

Definitions
-----------

1. Is the observable variable :math:`x(t)` defined clearly enough that a
   reader could compute it from embeddings?
2. Is the difference between semantic embedding space and 4D engine space
   explicit?
3. Is "mantissa" defined before it is used?
4. Is the Kolmogorov-Smirnov statistic defined with its null distribution?
5. Is the distinction between human label, AI label, and actual authorship
   clear?

Mixture Model
-------------

1. Can a reader derive the direct AI score from the two Gaussian
   likelihoods?
2. Is orientation explained as held-out AUC behavior rather than arbitrary
   score flipping?
3. Is the expert weight formula complete: marginal likelihood, competence,
   reliability, and temperature?
4. Are the expert-disagreement rules defined without implying that they
   are posterior probabilities?

Geometry
--------

1. Is the context plane defined?
2. Is plane alignment distinguished from plane residual?
3. Is geometric confidence defined as a reliability proxy, not a posterior?
4. Is it clear why an expert can be confident but geometrically
   incompetent?

Evidence
--------

1. Are hard, soft, and ambiguous evidence defined by thresholds?
2. Is it clear why ambiguous rows do not contribute directional weighted
   evidence?
3. Is the document label clearly derived from local windows/passages?
4. Is the whole-document posterior kept separate from document evidence?

Failure Modes
-------------

1. Does the documentation explain why some text cannot be reliably
   separated?
2. Does it explain the Moby-Dick-style contradiction between posterior
   lean and local evidence?
3. Does it warn against treating one threshold as the model?
4. Does it require ROC/AUC and coverage reporting alongside accuracy?

Reader Challenge
----------------

After reading these docs, a reviewer should be able to answer:

* What exactly is measured?
* What exactly is modeled?
* What exactly is exposed to users?
* What is hidden?
* What does confidence mean?
* When should the engine refuse a hard directional claim?
* Which source files implement each concept?

If any answer requires guessing from code rather than reading the docs,
the docs need another pass.
