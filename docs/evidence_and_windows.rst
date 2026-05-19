Evidence, Windows, And Documents
================================

Why Windows Matter
------------------

Documents are mixtures of local regimes. Human text can contain AI-like
passages, and AI text can contain human-like passages. Some template-like
text may be indistinguishable under the current measurement.

For that reason, the browser/API path scores:

* provided passages;
* overlapping word windows at 80, 120, and 240 words;
* the whole text.

Each row is evidence, not a verdict. The document result is an aggregate
over rows, with the whole-text posterior kept as a separate diagnostic.

Window stride is half the window length. The server caps each scale to
the first twelve windows for interactive latency.

This design is intentionally different from a one-number detector. A
document-level average can hide the structure that matters: one hard AI
window, many ambiguous windows, or a weak posterior lean with no reliable
local support. Arepo exposes those cases separately.

Evidence Classes
----------------

For a local score, define:

.. math::

   M = 2 |p_{AI} - 0.5|

and let :math:`C_g` be geometric confidence.

Current public thresholds are:

.. code-block:: text

   ambiguous_margin = 0.10
   ambiguous_geometric_confidence = 0.10
   hard_margin = 0.25
   hard_geometric_confidence = 0.25

The local evidence class is:

.. list-table::
   :header-rows: 1

   * - Condition
     - Evidence class
   * - :math:`M \le 0.10` or :math:`C_g \le 0.10`
     - ``ambiguous``
   * - :math:`M \ge 0.25`, :math:`C_g \ge 0.25`, and :math:`p_{AI} \ge 0.5`
     - ``hard_ai``
   * - :math:`M \ge 0.25`, :math:`C_g \ge 0.25`, and :math:`p_{AI} < 0.5`
     - ``hard_human``
   * - otherwise, with :math:`p_{AI} \ge 0.5`
     - ``soft_ai``
   * - otherwise, with :math:`p_{AI} < 0.5`
     - ``soft_human``

Document Aggregation
--------------------

Document evidence is computed from local rows. Window rows are used when
available; otherwise passage rows are used. The summary includes:

* mean, median, minimum, maximum, and variance of :math:`p_{AI}`;
* fraction of hard AI windows;
* fraction of hard human windows;
* fraction of ambiguous windows;
* soft evidence balance;
* longest hard-AI and hard-human runs;
* strongest hard-AI and hard-human windows;
* margin-weighted side evidence;
* geometric-confidence-weighted side evidence;
* scale consistency across window sizes.

Ambiguous rows are excluded from side-weighted evidence. If all rows are
ambiguous, margin-weighted and geometric-weighted AI evidence are both
reported as 0.5.

Document Evidence Labels
------------------------

The document label is based on accumulated local evidence:

.. list-table::
   :header-rows: 1

   * - Local evidence pattern
     - Document label
   * - both hard AI and hard human evidence are present
     - Mixed hard evidence
   * - any hard AI evidence and no hard human evidence
     - Hard AI evidence
   * - any hard human evidence and no hard AI evidence
     - Hard Human evidence
   * - at least 80 percent ambiguous rows
     - Ambiguous evidence
   * - soft evidence balance at least 0.20
     - Soft AI evidence
   * - soft evidence balance at most -0.20
     - Soft Human evidence
   * - otherwise
     - Mixed soft evidence

The whole-document posterior is still returned as ``posterior_label`` and
``posterior_verdict``. It is not allowed to override the local evidence
summary in the main document label.

The confidence layer is what prevents false precision. A local row must
have both posterior margin and geometric confidence before it becomes hard
evidence. Without that support, it remains soft or ambiguous even if the
posterior leans toward AI.

Example: Moby-Dick
------------------

The Moby-Dick demo can produce a narrow whole-document AI lean while all
local windows are ambiguous. The document evidence label is therefore
"Ambiguous evidence". The posterior lean remains visible as a separate
score.

Example: Declaration Of Independence
------------------------------------

Historic civic prose can be formal, compressed, and rhetorically regular.
Those properties can make some passages look close to generated text in a
raw posterior. Arepo should not convert that proximity into a hard AI
claim unless local windows also pass the confidence requirements. The
expected behavior is to show the lean, show the confidence, and avoid a
hard AI label when the support is weak.
