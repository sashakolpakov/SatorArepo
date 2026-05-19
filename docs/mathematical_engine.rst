Mathematical Engine
===================

The Core Observable
-------------------

Let :math:`t` be a text segment and let :math:`M_j` be one transformer
model from the production model list. The model produces an embedding
array:

.. math::

   E_j(t) \in \mathbb{R}^{n_j \times d_j}.

The implementation flattens this array into a vector of real values:

.. math::

   z_{j,1}, z_{j,2}, \ldots, z_{j,N_j}.

Zeros, infinities, and NaNs are discarded. For each remaining value, the
engine computes the base-10 mantissa:

.. math::

   m_{j,k} = \{\log_{10} |z_{j,k}|\},

where :math:`\{x\}` is the fractional part. Thus
:math:`m_{j,k} \in [0,1)`.

Why Mantissas?
--------------

The mantissa distribution is a scale-free view of embedding values.
For a roughly scale-invariant positive variable, base-10 mantissas are
approximately uniform on :math:`[0,1)`.

The engine does not assume that all embeddings should be exactly
Benford. It uses deviations from mantissa uniformity as a low-dimensional
statistical signature. The assumption is weaker:

   Human and LLM text can induce different deviations from mantissa
   uniformity in different transformer model classes.

For one model :math:`M_j`, let :math:`F_{j,t}` be the empirical CDF of
the mantissas extracted from :math:`t`. The production statistic is the
Kolmogorov-Smirnov distance from uniform:

.. math::

   D_j(t) = \sup_{u \in [0,1]} |F_{j,t}(u) - u|.

Under the ideal null of independent uniform mantissas,
:math:`\sqrt{n}D_j` follows the Kolmogorov limiting law. The engine uses
:math:`D_j` as a feature, not as a literal p-value, because embedding
coordinates are dependent and model-specific.

The 4D feature vector is:

.. math::

   x(t) =
   \left(
   D_1(t), D_2(t), D_3(t), D_4(t)
   \right) \in \mathbb{R}^4.

This vector is the observable passed to the classifier. The original
semantic embeddings are only the source from which the four distributional
statistics are computed.

Generative Vs Masked Model Asymmetry
------------------------------------

The production feature vector uses two generative models and two masked
or discriminative models. In code:

.. code-block:: text

   Generative:
     EleutherAI/gpt-neo-125m
     flax-community/gpt-neo-125M-code-clippy

   Masked/discriminative:
     distilbert-base-uncased
     google/electra-base-discriminator

For generative models, the implementation uses token embedding weights
when available. For masked models, it uses the final hidden state. The two
model classes expose different internal views of the same text.

Long Text Chunking
------------------

Transformer models have token limits. For long text, the implementation
splits tokens into overlapping chunks:

.. math::

   [0,L), [L/2, 3L/2), [L, 2L), \ldots

where :math:`L` is the model-specific maximum length. The default is
1024 tokens for generative models and 512 tokens for masked models.

The KS statistic is computed per chunk and averaged:

.. math::

   D_j(t) = \frac{1}{K} \sum_{k=1}^K D_{j,k}(t).

This avoids scoring only the prefix of a long document.

Auxiliary Statistics
--------------------

The code also contains an auxiliary multi-statistic extraction path:

* Kolmogorov-Smirnov statistic;
* Cramer-von Mises statistic;
* Anderson-Darling statistic;
* Benford first-digit MSE;
* Benford first-digit KL divergence.

The current production MoE path uses the 4D KS vector, not the
multi-statistic vector.
