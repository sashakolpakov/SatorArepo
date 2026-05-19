Glossary
========

4D engine space
   The four-dimensional vector of mantissa KS statistics, one coordinate
   per transformer model.

AI label
   In training records, ``label=1``. It means generated or AI-source text
   for that dataset, not guaranteed authorship.

Authorship
   The real origin of the text. It can differ from a dataset label when a
   source is noisy or when text is templated enough to be ambiguous.

Ambiguous evidence
   A local or document result whose posterior margin or geometric support
   is too weak for a hard directional claim.

Confidence level
   The public reliability level attached to a score. In Arepo, confidence
   is driven by posterior margin and geometric support; it is not the same
   thing as the AI probability.

Benford-style statistic
   A statistic based on leading digits or mantissas of real-valued model
   outputs. The production feature is a mantissa-uniformity KS statistic.

Context geometry
   The pooled mean, standard deviation, and PCA basis fitted to an
   expert's source context.

Context plane
   The low-dimensional PCA plane stored in an expert context model.

Diagonal Gaussian
   A Gaussian density whose covariance matrix is assumed diagonal. Each
   coordinate has its own mean and standard deviation.

Evidence class
   One of ``hard_ai``, ``hard_human``, ``soft_ai``, ``soft_human``, or
   ``ambiguous``.

Geometric confidence
   A reliability proxy equal to posterior margin times aligned majority
   expert-weight gap.

Hard evidence
   Evidence with enough posterior margin and geometric confidence to pass
   the hard gate.

Human label
   In training records, ``label=0``.

Kolmogorov-Smirnov statistic
   The maximum distance between an empirical CDF and a reference CDF. Here
   the reference is uniform mantissas on :math:`[0,1)`.

Kolmogorov null
   The ideal independent-uniform reference law for :math:`\sqrt{n}D`.
   The engine uses the KS value as a feature, not as a strict p-value.

Mantissa
   The fractional part of :math:`\log_{10}|x|`, lying in :math:`[0,1)`.

Mixture of experts
   A set of local Gaussian experts whose oriented scores are combined by
   likelihood, context competence, optional reliability, and temperature.

Oriented AI score
   The expert AI score after correcting for learned regime orientation.

Posterior margin
   :math:`2 |p_{AI} - 0.5|`. A margin of 0 means exactly 50/50; a margin
   of 1 means an extreme posterior.

Posterior label
   The side with the larger posterior score. It is a diagnostic lean and
   should not be treated as the public verdict when confidence or local
   evidence is weak.

Regime inversion
   A dataset or source context where the direct likelihood ratio ranks
   classes backward on held-out data and must be oriented.

Soft evidence
   Directional evidence that does not pass the hard gate.

Window
   A word-based segment scored separately from whole passages. The server
   uses 80, 120, and 240 word windows with 50 percent overlap.
