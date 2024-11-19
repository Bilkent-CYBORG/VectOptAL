.. role:: hidden
    :class: hidden-section

vopy.acquisition
===================================

.. automodule:: vopy.acquisition
.. currentmodule:: vopy.acquisition

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
-----------------------------------

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
-----------------------------------------

.. autofunction:: optimize_acqf_discrete

.. autofunction:: optimize_decoupled_acqf_discrete
