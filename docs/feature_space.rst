The 4D Feature Space
====================

Coordinates
-----------

Every scored text unit becomes a point:

.. math::

   x = (x_1, x_2, x_3, x_4),

where each coordinate is a mantissa-uniformity KS statistic from one
transformer model. This is the "4D engine space" used throughout the
project.

Interpretation
--------------

The coordinates are not semantic embeddings. They are distributional
statistics of embedding values. This distinction is essential:

* an embedding vector says something about content and context;
* the 4D engine vector says something about the distributional shape of
  model activations or embeddings induced by the text.

The current mixture-of-experts model operates only in this 4D space.

Feature Geometry
----------------

Experts learn two kinds of geometry:

Class geometry
   Diagonal Gaussian models for human and AI samples.

Context geometry
   A pooled source-context model fitted to the expert's training data.
   This context model checks whether the input is near the expert's
   training regime.

For a training subset :math:`S_j`, the context model stores:

.. math::

   \mu_{C,j},\quad \sigma_{C,j},\quad U_j.

Here :math:`\mu_{C,j}` is the coordinate mean, :math:`\sigma_{C,j}` is
the coordinate standard deviation, and :math:`U_j` is a PCA basis for a
low-dimensional context plane in standardized coordinates.

The context plane is fitted from pooled human and AI training points for
that expert. It is a plane in 4D feature space, not a semantic embedding
plane.

Standardized Context Coordinates
--------------------------------

For an input point :math:`x`, expert :math:`j` computes:

.. math::

   z_j = \frac{x - \mu_{C,j}}{\sigma_{C,j}}.

The context distance is the root-mean-square standardized distance:

.. math::

   r_j(x) = \sqrt{\frac{1}{d}\sum_{k=1}^{d} z_{j,k}^2}.

The context-plane projection is:

.. math::

   \hat{z}_j = U_j U_j^\top z_j.

The residual is:

.. math::

   \rho_j(x) = \|z_j - \hat{z}_j\|_2.

The plane alignment is:

.. math::

   a_j(x) =
   \begin{cases}
   1, & \|z_j\|_2 \le \epsilon,\\
   \|\hat{z}_j\|_2 / \|z_j\|_2, & \text{otherwise.}
   \end{cases}

High alignment means the point lies mostly in the training context plane.
Low residual means the point is near the plane.

These are related but not identical. Alignment is directional; residual
is distance from the plane.

Why Context Geometry Is Needed
------------------------------

An expert can have a strong likelihood ratio and still be wrong outside
its training regime. Context geometry asks:

   Is this expert being asked about a point near the kind of points it
   learned from?

This is separate from the class posterior. A model can be confident in
posterior direction but geometrically unsupported.
