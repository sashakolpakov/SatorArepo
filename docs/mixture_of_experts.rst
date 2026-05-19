Mixture Of Experts
==================

Expert Model
------------

Each expert :math:`j` is trained on a named source group or local subset.
For human samples it fits a diagonal Gaussian:

.. math::

   p_j(x \mid H) =
   \prod_{k=1}^{d}
   \mathcal{N}(x_k; \mu_{H,j,k}, \sigma_{H,j,k}^2).

For AI samples it fits:

.. math::

   p_j(x \mid A) =
   \prod_{k=1}^{d}
   \mathcal{N}(x_k; \mu_{A,j,k}, \sigma_{A,j,k}^2).

The implementation regularizes standard deviations with a floor and an
inflation factor to avoid degenerate densities.

Direct AI Score
---------------

For expert :math:`j`, define log likelihoods:

.. math::

   \ell_{H,j}(x) = \log p_j(x \mid H),
   \qquad
   \ell_{A,j}(x) = \log p_j(x \mid A).

The direct AI score is:

.. math::

   q_j(x) =
   \sigma(\ell_{A,j}(x) - \ell_{H,j}(x)),

where :math:`\sigma(u) = 1/(1 + e^{-u})`.

Orientation
-----------

Some datasets exhibit regime inversion: the direct likelihood ratio may
rank AI and human in the wrong direction for that local source. The
engine learns an orientation on held-out data:

.. math::

   o_j =
   \begin{cases}
   +1, & \text{if held-out AUC} \ge 0.5,\\
   -1, & \text{if held-out AUC} < 0.5.
   \end{cases}

The oriented AI score is:

.. math::

   s_j(x) =
   \begin{cases}
   q_j(x), & o_j = +1,\\
   1 - q_j(x), & o_j = -1.
   \end{cases}

Orientation is learned per expert from held-out source behavior.

Expert Weighting
----------------

Each expert contributes a class-agnostic marginal likelihood:

.. math::

   m_j(x) =
   \log\left(
   \frac{
   \exp(\ell_{H,j}(x)) + \exp(\ell_{A,j}(x))
   }{2}
   \right).

With the production ``plane`` competence metric, context competence is:

.. math::

   c_j(x) = \exp\left(-\frac{1}{2}\rho_j(x)^2\right),

where :math:`\rho_j(x)` is context-plane residual.

If reliability calibration is present, the expert also has
:math:`R_j(x)`, a held-out estimate of correctness for the expert's
predicted side and context-distance bin. Without a reliability table,
:math:`R_j(x)=1`.

The unnormalized log weight is:

.. math::

   g_j(x) =
   m_j(x)
   + \lambda_c \log c_j(x)
   + \lambda_r \log R_j(x).

The production web settings use:

.. code-block:: text

   temperature = 2.0
   competence_metric = plane
   competence_strength = 1.0
   alignment_threshold = 0.8

The normalized weight is a softmax:

.. math::

   w_j(x) =
   \frac{\exp(g_j(x)/T)}
        {\sum_k \exp(g_k(x)/T)}.

Mixture Posterior
-----------------

The mixture AI probability is:

.. math::

   p_{AI}(x) = \sum_j w_j(x) s_j(x).

The human probability is:

.. math::

   p_H(x) = 1 - p_{AI}(x).

The base prediction is AI if :math:`p_{AI}(x) \ge 0.5`, otherwise human.

Expert Disagreement Rules
-------------------------

The production settings include asymmetric expert-disagreement recovery:

.. code-block:: text

   ai_veto_threshold = 0.89
   ai_veto_min_weight = 0.20
   human_veto_threshold = 0.95
   human_veto_min_weight = 0.20

If the mixture prediction is human but a sufficiently weighted expert
strongly scores AI, the final prediction can switch to AI. The same rule
exists for strong human evidence.

The disagreement flags are decision rules. They are not extra posterior
probabilities and they do not change the meaning of ``p_ai``.

Geometric Confidence
--------------------

Geometric confidence is not a posterior. It is a reliability proxy derived
from how much aligned expert mass supports the winning side.

Let :math:`b` be the final predicted side. Define the posterior margin:

.. math::

   M(x) = 2 |p_{AI}(x) - 0.5|.

For each expert, determine whether its oriented score is on the same side
as :math:`b`. Also determine whether the expert is plane-aligned:

.. math::

   a_j(x) \ge \alpha,

where production :math:`\alpha = 0.8`.

Let :math:`W_+` be the total weight of aligned experts supporting the
winning side, and :math:`W_-` the total weight of aligned experts on the
opposite side. The aligned gap is:

.. math::

   G(x) = \max(0, W_+ - W_-).

The geometric confidence is:

.. math::

   C_g(x) = M(x) G(x).

The system treats this as support for the reliability of a local score,
not as the probability that the text is actually human or AI.

A high posterior with low :math:`C_g` is directional but weakly supported
by context geometry.
