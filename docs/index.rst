Arepo Mathematical Engine
================================

Arepo is an authorship-evidence engine. It does not try to
read a text and infer intention directly. It maps text into a small
observable statistical space, scores that point with a mixture of
locally competent experts, and reports the resulting evidence as
hard, soft, or ambiguous.

Arepo is confidence-first. It does not turn every posterior lean into a
claim of authorship. A text can be more AI-like than human-like in one
score and still be weak evidence if the supporting experts are not
geometrically aligned with that local context.

The important design rule is:

   Per-window scores are evidence, not verdicts. A document result is
   an accumulation of local evidence, not just the mean of one posterior.

This rule is meant to prevent a common failure mode of binary AI
detectors: confidently calling historic, formal, or public-domain human
prose AI-generated. If a Declaration of Independence paragraph lies near
generated prose in the observable space but lacks reliable local support,
Arepo should expose that uncertainty rather than issue a hard AI verdict.

This documentation is written for two audiences:

* maintainers who need to know exactly what the engine computes;
* reviewers who need to decide whether every mathematical notion is
  defined well enough to challenge, reproduce, or improve the system.

.. toctree::
   :maxdepth: 2
   :caption: Engine

   overview
   method
   mathematical_engine
   feature_space
   mixture_of_experts
   evidence_and_windows

.. toctree::
   :maxdepth: 2
   :caption: Product And Operations

   ui_api
   training_and_evaluation
   limitations
   experiment_lineage
   glossary
   authors
   publishing
   api
   socratic_review

Source Map
----------

The production path is concentrated in these modules:

.. list-table::
   :header-rows: 1

   * - Concept
     - Implementation
   * - 4D mantissa feature extraction
     - ``src/arepo/core.py`` and ``src/arepo/stats.py``
   * - Oriented Gaussian mixture of experts
     - ``src/arepo/mixture_experts.py``
   * - Public evidence classes and document aggregation
     - ``src/arepo/evidence.py`` and ``src/arepo/calibration.py``
   * - Browser/API serving path
     - ``src/arepo/web.py`` and ``src/arepo/templates/index.html``
   * - Built-in demo corpus
     - ``src/arepo/corpus.py``

Build
-----

Build the docs locally with:

.. code-block:: bash

   python3 -m sphinx -b html docs docs/_build/html

The generated site is written to ``docs/_build/html``.
