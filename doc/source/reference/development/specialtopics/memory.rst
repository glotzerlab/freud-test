=================
Memory Management
=================

Memory handling in **freud** is a somewhat intricate topic.
Most **freud** developers do not need to be aware of such details; however, certain practices must be followed to ensure that the expected behavior is achieved.
This page provides an overview of how data should be handled in **freud** and how module developers should use **freud**'s core classes to ensure proper memory management.
A thorough description of the process is also provided for developers who need to understand the internal logic for further development.

.. note::

    This page specifically deals with modules primarily written in C++. These
    concepts do not apply to pure Python/Cython modules.

Problem Statement
=================

The standard structure for **freud** modules involves a core implementation in a C++ class wrapped in a Cython class that owns a pointer to the C++ object.
Python ``compute`` methods call through to C++ ``compute`` methods, which perform the calculation and populate class member arrays that are then accessed via properties of the owning Cython class.
These classes are designed to be reusable, i.e. ``compute`` may be called many times on the same object with different data, and the accessed properties will return the most current data.
Users have a reasonable expectation that if the accessed property is saved to another variable it will remain unchanged by future calls to ``compute`` or if the originating C++ object is destructed, but a naive implementation that ensures this invariant would involve reallocating memory on every call to compute, an unnecessarily expensive operation.
Ultimately, what we want is a method that performs the minimal number of memory allocations while allowing users to operate transparently on outputs without worrying about whether the data will be invalidated by future operations.

ManagedArray
============

The **freud** ``ManagedArray`` template class provides a solution to this problem for arbitrary types of numerical data.
Proper usage of the class can be summarized by the following steps:

#. Declaring ``ManagedArray`` class members in C++.
#. Calling the ``prepare`` method in every ``compute``.
#. Making the array accessible via a getter method that **returns a const reference**.
#. Calling ``make_managed_numpy_array`` in Cython and returning the output as a property.

Plenty of examples of following this pattern can be found throughout the codebase, but for clarity we provide a complete description with examples below.
If you are interested in more details on the internals of ``ManagedArray`` and how it actually works, you can skip to :ref:`managedarray_explained`.


Using ManagedArrays
-------------------

We'll use :mod:`freud.cluster.Cluster` to illustrate how the four steps above may be implemented.
This class takes in a set of points and assigns each of them to clusters, which are store in the C++ array ``m_cluster_idx``.

Step 1 is simple: we note that ``m_cluster_idx`` is a member variable of type ``ManagedArray<unsigned int>``.
For step 2, we look at the first few lines of ``Cluster::compute``, where we see a call to ``m_cluster_idx.prepare``.
This method encapsulates the core logic of ``ManagedArray``, namely the intelligent reallocation of memory whenever other code is still accessing it.
This means that, if a user saves the corresponding Python property :attr:`freud.cluster.Cluster.cluster_idx` to a local variable in a script and then calls :attr:`freud.cluster.Cluster.compute`, the saved variable will still reference the original data, and the new data may be accessed again using :attr:`freud.cluster.Cluster.cluster_idx`.

Step 3 for the cluster indices is accomplished in the following code block:

.. code-block:: C++

    //! Get a reference to the cluster ids.
    const util::ManagedArray<unsigned int> &getClusterIdx() const
    {
        return m_cluster_idx;
    }

**The return type of this method is crucial: all such methods must return const references to the members**.

The final step is accomplished on the Cython side.
Here is how the cluster indices are exposed in :class:`freud.cluster.Cluster`:

.. code-block:: python

    @_Compute._computed_property
    def cluster_idx(self):
        """:math:`N_{points}` :class:`numpy.ndarray`: The cluster index for
        each point."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterIdx(),
            freud.util.arr_type_t.UNSIGNED_INT)

Essentially all the core logic is abstracted away from the user through the :func:`freud.data.make_managed_numpy_array`, which creates a NumPy array that is a view on an existing ``ManagedArray``.
This NumPy array will, in effect, take ownership of the data in the event that the user keeps a reference to it and requests a recomputation.
Note the signature of this function: the first argument must be **a pointer to the ManagedArray** (which is why we had to return it by reference), and the second argument indicates the type of the data (the possible types can be found in ``freud/util.pxd``).
There is one other point to note that is not covered by the above example; if the template type of the ``ManagedArray`` is not a scalar, you also need to provide a third argument indicating the size of this vector.
The most common use-case is for methods that return an object of type ``ManagedArray<vec3<float>>``: in this case, we would call ``make_managed_numpy_array(&GETTER_FUNC, freud.util.arr_type_t.FLOAT, 3)``.


Indexing ManagedArrays
----------------------

With respect to indexing, the ``ManagedArray`` class behaves like any standard array-like container and can be accessed using e.g. ``m_cluster_idx[index]``.
In addition, because many calculations in **freud** output multidimensional information, ``ManagedArray`` also supports multidimensional indexing using ``operator()``.
For example, setting the element at second row and third column of a 2D ``ManagedArray`` ``array`` to one can be done using ``array(1, 2) = 1`` (indices beginning from 0).
Therefore, ``ManagedArray`` objects can be used easily inside the core C++ calculations in **freud**.


.. _managedarray_explained:

Explaining ManagedArrays
------------------------

We now provide a more detailed accounting of how the ``ManagedArray`` class actually works.
Consider the following block of code:

.. code-block:: python

    rdf = freud.density.RDF(bins=100, r_max=3)

    rdf.compute(system=(box1, points1))
    rdf1 = rdf.rdf

    rdf2.compute(system=(box2, points2))
    rdf2 = rdf.rdf

We require that ``rdf1`` and ``rdf2`` be distinct arrays that are only equal if the results of computing the RDF are actually equivalent for the two systems, and we want to achieve this with the minimal number of memory allocations.
In this case, that means there are two required allocations; returning copies would double that.

To achieve this goal, ``ManagedArray`` objects store a pointer to a pointer.
Multiple ``ManagedArray`` objects can point to the same data array, and the pointers are all shared pointers to automate deletion of arrays when no references remain.
The key using the class properly is the ``prepare`` method, which checks the reference count to determine whether it's safe to simply zero out the existing memory or if it needs to allocate a new array.
In the above example, when ``compute`` is called a second time the ``rdf1`` object in Python still refers to the computed data, so ``prepare`` will detect that there are multiple (two) shared pointers pointing to the data and choose to reallocate the class's ``ManagedArray`` storing the RDF.
By calling ``prepare`` at the top of every ``compute`` method, developers ensure that the array used for the rest of the method has been properly zeroed out, and they do not need to worry about whether reallocation is needed (including cases where array sizes change).

To ensure that all references to data are properly handled, some additional logic is required on the Python side as well.
The Cython ``make_managed_numpy_array`` instantiates a ``_ManagedArrayContainer`` class, which is essentially just a container for a ``ManagedArray`` that points to the same data as the ``ManagedArray`` provided as an argument to the function.
This link is what increments the underlying shared pointer reference counter.
The ``make_managed_numpy_array`` uses the fact that a ``_ManagedArrayContainer`` can be transparently converted to a NumPy array that points to the container; as a result, no data copies are made, but all NumPy arrays effectively share ownership of the data along with the originating C++ class.
If any such arrays remain in scope for future calls to ``compute``, ``prepare`` will recognize this and reallocate memory as needed.
