.. role:: hidden
    :class: hidden-section

vectoptal.acquisition
===================================

.. automodule:: vectoptal.acquisition
.. currentmodule:: vectoptal.acquisition

.. autoclass:: AcquisitionStrategy
    :members:


Acquisiton Strategies
-----------------------------

:hidden:`SumVarianceAcquisition`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SumVarianceAcquisition
    :members:

:hidden:`MaxDiagonalAcquisition`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MaxDiagonalAcquisition
    :members:


Decoupled Acquisiton Strategies
-----------------------------

.. autoclass:: DecoupledAcquisitionStrategy
    :members:

:hidden:`MaxVarianceDecoupledAcquisition`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MaxVarianceDecoupledAcquisition
    :members:

:hidden:`ThompsonEntropyDecoupledAcquisition`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ThompsonEntropyDecoupledAcquisition
    :members:

Utilities for Acquisiton Strategies
-----------------------------

.. autofunction:: optimize_acqf_discrete

.. autofunction:: optimize_decoupled_acqf_discrete
