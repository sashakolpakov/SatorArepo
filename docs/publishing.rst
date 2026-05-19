Publishing
==========

Local Build
-----------

The docs are ordinary Sphinx documentation. Build them with:

.. code-block:: bash

   python3 -m sphinx -W -b html docs docs/_build/html

or:

.. code-block:: bash

   make -C docs html

The ``-W`` flag treats warnings as errors. Use it before committing docs
changes.

GitHub Pages Workflow
---------------------

The repository includes ``.github/workflows/docs.yml``. The workflow:

* installs the project with the ``docs`` optional dependencies;
* builds the Sphinx HTML site;
* uploads the built site as a Pages artifact on ``main``;
* deploys to GitHub Pages on ``main``.

The workflow also builds on pull requests to ``main`` so broken docs are
caught before merge.

Server Smoke Test
-----------------

The CI workflow starts the Flask app as a short-lived process and checks
``/health`` plus ``/demo-corpus``. It disables transformer preloading for
this smoke test, so it verifies packaging, routing, templates, and the
bundled MoE model boundary without downloading large model weights.

GitHub Actions is not a public hosting target for the workbench. Jobs run
on temporary runners and are intended for build, test, and deployment
automation. Use Actions to test or deploy the server; host the live app on
a real runtime.

Publication Boundary
--------------------

Generated experiment artifacts are not documentation by default.
``reports/`` and ``visualizations/`` should remain uncommitted unless a
specific report is intentionally cited by the docs or used as
reproducibility evidence.

When a report is cited, the docs should state:

* what dataset or guardrail suite was used;
* which model file was evaluated;
* which threshold or gating rule was applied;
* what residual risk remains.
