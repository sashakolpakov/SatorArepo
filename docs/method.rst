Method
======

Arepo detects AI-generated text by analysing the statistical
properties of transformer embedding values.

Mantissa Extraction
-------------------

For each embedding value :math:`x`, compute the fractional part of
its base-10 logarithm:

.. math::

   m = \log_{10}|x| \bmod 1

By the generalised Benford's Law, values drawn from a broad
log-scale distribution have mantissas approximately uniformly
distributed on :math:`[0, 1)`.

KS Test
-------

A Kolmogorov-Smirnov test compares the empirical mantissa CDF to
:math:`\mathrm{Uniform}(0,1)`.  The KS statistic :math:`D_n` measures
the maximum deviation.

4-D Feature Space
-----------------

Each text is embedded by four transformer models (two generative, two
masked).  The KS statistic from each model gives a 4-dimensional
feature vector :math:`\mathbf{f} \in \mathbb{R}^4`.

Mixture Of Oriented Experts
---------------------------

The production engine is a mixture of oriented Gaussian experts.
Each expert fits one diagonal Gaussian model for human samples and one
for AI samples in the 4D feature space. The expert's direct AI score is
the logistic transform of the AI log likelihood minus the human log
likelihood.

Each expert also learns whether the direct likelihood ratio should be
used as-is or inverted. That orientation is determined from held-out AUC
inside the expert's training regime.

Expert opinions are combined by a softmax over marginal likelihood,
context-plane competence, optional held-out reliability, and temperature.
The public result is then converted into hard, soft, or ambiguous
evidence using posterior margin and geometric confidence.

The confidence conversion is part of the method. Arepo does not present
the mixture posterior as a standalone verdict. A posterior lean without
geometric support is a weak signal, not a hard authorship claim.

Generative vs Masked Asymmetry
-------------------------------

Generative and masked models show **opposite** KS patterns for human
vs AI text.  On generative models, human embeddings deviate *more* from
Benford uniformity (higher KS); on masked models, human embeddings
deviate *less* (lower KS).  This asymmetry creates a diagonal
separation in the 2-D (generative-KS, masked-KS) feature space.

Reference
---------

This codebase was originally motivated by Benford-style embedding
statistics. The current production path is the 4D mantissa-KS MoE path
described in :doc:`mathematical_engine` and :doc:`mixture_of_experts`.

The practical purpose of the MoE confidence layer is to avoid forced
binary labels in regimes where the observable distributions overlap. This
is especially important for formal, historical, public-domain, or
template-like text, where a hard AI accusation from a weak score is a
modeling error rather than useful evidence.
