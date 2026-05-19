UI And API
==========

Public Boundary
---------------

The browser UI exposes public scores and evidence summaries. It does not
expose internal expert names, expert votes, raw likelihoods, or model
geometry.

The UI is confidence-first. It shows how strong the evidence is, not just
which side has the larger posterior. A low-confidence AI lean is not
displayed as a hard AI finding.

Routes
------

``GET /``
   Serves the browser workbench.

``GET /demo-corpus``
   Returns built-in demo entries. The response uses
   ``Cache-Control: no-store`` so the browser does not keep stale demo
   entries after local changes.

``POST /analyze``
   Scores submitted text. Accepts either:

   .. code-block:: json

      {"text": "paragraphs separated by blank lines"}

   or:

   .. code-block:: json

      {
        "passages": [
          {"title": "Passage 1", "text": "..."}
        ]
      }

``GET /health``
   Reports server status, including whether the MoE model and transformer
   models are loaded.

Public Result Shape
-------------------

The response contains:

``document``
   Document-level evidence label, posterior scores, evidence summary,
   rankings, and scale consistency.

``passages``
   Public score objects for each supplied or inferred passage.

``windows``
   Public score objects for 80/120/240 word windows.

Important public fields include:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``p_ai`` / ``p_human``
     - Mixture posterior scores.
   * - ``posterior_margin``
     - :math:`2 |p_{AI} - 0.5|`.
   * - ``geometric_confidence``
     - Context-aligned majority support times posterior margin.
   * - ``evidence_class``
     - Local evidence class.
   * - ``hard_gate_accepted``
     - Whether the local row is hard evidence.
   * - ``document_evidence_label``
     - Document label from accumulated local evidence.
   * - ``posterior_label``
     - Whole-document posterior label, kept separate from evidence label.

Consumers should prefer ``document_evidence_label`` for public
interpretation. ``posterior_label`` is a diagnostic lean. It is useful
for calibration and ROC analysis, but it does not replace confidence,
hard-evidence coverage, and local window support.

Demo Corpus
-----------

The browser demo currently includes:

* Declaration of Independence: 3 human public-domain passages.
* Moby-Dick: 4 human public-domain passages.
* Jane Eyre: 4 human public-domain passages.
* LLM Control: Soft/Ambiguous: 4 generated passages whose document result
  is ambiguous or weak evidence.
* LLM Control: Hard Evidence: 1 generated passage that produces hard AI
  evidence under the current model.

Running Locally
---------------

.. code-block:: bash

   python3 -m arepo.web --host 127.0.0.1 --port 5000

The default server loads the bundled MoE model and the four transformer
models used by the 4D engine.
