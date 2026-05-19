Limitations And Design Consequences
===================================

Overlap Is Real
---------------

If human and AI distributions overlap in the 4D observable space, no
threshold can extract reliable certainty from that projection. The engine
must distinguish:

Indistinguishable in reality
   The text is formulaic or constrained enough that authorship may not be
   inferable from the text alone.

Indistinguishable under current measurement
   The current 4D mantissa feature map fails to separate the source, even
   though another observable might.

The product response is the same at the point of scoring: weak or
ambiguous evidence.

The system should not pretend otherwise. A detector that always returns a
hard label will eventually convert overlap into false certainty. That is
the failure mode behind many high-profile mistakes on historic,
public-domain, and encyclopedia-like human prose.

Confidence Is A Primary Output
------------------------------

Arepo reports confidence because posterior direction alone is not enough.
A score such as 55 percent AI can mean either "weak AI-leaning evidence"
or "hard AI evidence", depending on local geometry and window support.
The public result therefore includes:

* posterior scores;
* geometric confidence;
* hard, soft, and ambiguous evidence classes;
* hard-evidence coverage;
* ambiguity coverage;
* document evidence labels separate from posterior labels.

This distinction is mandatory for safety. The Declaration of Independence
should not become "AI-generated" because one averaged posterior leans
AI-like. If the local windows are weak or geometrically unsupported, the
correct result is weak or ambiguous evidence.

Why Hard Evidence Has Lower Coverage
------------------------------------

Hard evidence requires both posterior margin and geometric confidence.
This improves reliability but lowers coverage. Report coverage alongside
accuracy.

Why More Features Are Not Automatically Better
----------------------------------------------

Adding covariates can improve performance, damage guardrails, or overfit
to a dataset. The production observable is the 4D feature space because
it is cheap, inspectable, and central to the MoE geometry.

Accept additional features only when they improve held-out guardrails
without breaking protected human text or known generated controls.

Known Operational Risks
-----------------------

* Short text often carries weak authorship evidence.
* Formal and prescriptive writing can be ambiguous.
* Classic or historical prose can be mapped near generated prose.
* Generated text can be human-like in the 4D feature space.
* Whole-document averaging can wash out strong local evidence.
* Local windows can contradict the whole-document posterior.
* A strong expert outside its context geometry can be unreliable.

Required Reporting Discipline
-----------------------------

Any evaluation report should separate:

* raw posterior scores;
* geometric confidence;
* hard-evidence coverage;
* ambiguity fraction;
* document-level aggregation behavior;
* false positives and false negatives by source regime;
* ROC/AUC behavior independent of one chosen threshold.

Branch Hygiene
--------------

Experiment branches can contain valuable reports without being useful
implementation branches. Before deleting such a branch, check whether it
contains:

* experimental reports not yet summarized in docs;
* generated models that are not reproducible elsewhere;
* guardrail findings that explain current thresholds;
* code that was never ported into the main implementation.

For ``math-evidence-windowing``, the source ideas have been absorbed or
superseded by the current sprint branch. The reports remain the part worth
preserving.
