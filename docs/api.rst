Implementation Map
==================

This page maps mathematical concepts to source modules.

Feature Extraction
------------------

``src/arepo/core.py``
   Defines the production model list and the 4D extraction entry points:
   ``extract_4d_features_with_indices`` and ``extract_4d_features``.

``src/arepo/stats.py``
   Implements mantissa extraction, chunked KS computation, auxiliary
   mantissa statistics, Gaussian fitting, and evaluation utilities.

Mixture Of Experts
------------------

``src/arepo/mixture_experts.py``
   Implements diagonal Gaussian experts, orientation learning, context
   geometry, reliability calibration, mixture prediction, expert
   disagreement rules, model serialization, and guardrail evaluation.

Evidence Layer
--------------

``src/arepo/evidence.py``
   Implements posterior margin, geometric-confidence evidence classes,
   document aggregation, scale consistency, and document evidence labels.

``src/arepo/calibration.py``
   Converts MoE scores into public score fields while requiring geometric
   confidence.

Serving Layer
-------------

``src/arepo/web.py``
   Loads the MoE model, preloads transformer models, scores passages and
   windows, and serves the public JSON API.

``src/arepo/templates/index.html``
   Renders the browser UI.

Data And Guardrails
-------------------

``src/arepo/training_data.py``
   Loads local and optional external training records.

``src/arepo/guardrail_evaluation.py``
   Builds guardrail records and writes evaluation reports.

``src/arepo/corpus.py``
   Defines the browser demo corpus.
