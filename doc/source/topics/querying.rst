.. _querying:

=========
Query API
=========

This page provides a thorough review of how neighbor finding is structured in **freud**.
It assumes knowledge at the level of the :ref:`neighbors` level of the tutorial; if you're not familiar with using the :meth:`query <freud.locality.NeighborQuery.query>` method with query arguments to find neighbors of points, please familiarize yourself with that section of the tutorial.

The central interface for neighbor finding is the :py:class:`freud.locality.NeighborQuery` family of classes, which provide methods for dynamically finding neighbors given a :py:class:`freud.box.Box`.
The :py:class:`freud.locality.NeighborQuery` class defines an abstract interface for neighbor finding that is implemented by its subclasses, namely the :py:class:`freud.locality.LinkCell` and :py:class:`freud.locality.AABBQuery` classes.
These classes represent specific data structures used to accelerate neighbor finding.
These two different methods have different performance characteristics, but in most cases :class:`freud.locality.AABBQuery` performs at least as well as, if not better than, :class:`freud.locality.LinkCell` and is entirely parameter free, so it is the default method of choice used internally in **freud**'s ``PairCompute`` classes.

In general, these data structures operate by constructing them using one set of points, after which they can be queried to efficiently find the neighbors of arbitrary other points using :py:meth:`freud.locality.NeighborQuery.query`.


Query Arguments
===============

The table below describes the set of valid query arguments.

+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+
| Query Argument | Definition                                                            | Data type | Legal Values              | Valid for                                                           |
+================+=======================================================================+===========+===========================+=====================================================================+
| mode           | The type of query to perform (distance cutoff or number of neighbors) | str       | 'none', 'ball', 'nearest' | :class:`freud.locality.AABBQuery`, :class:`freud.locality.LinkCell` |
+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+
| r_max          | Maximum distance to find neighbors                                    | float     | r_max > 0                 | :class:`freud.locality.AABBQuery`, :class:`freud.locality.LinkCell` |
+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+
| r_min          | Minimum distance to find neighbors                                    | float     | 0 <= r_min < r_max        | :class:`freud.locality.AABBQuery`, :class:`freud.locality.LinkCell` |
+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+
| num_neighbors  | Number of neighbors                                                   | int       | num_neighbors > 0         | :class:`freud.locality.AABBQuery`, :class:`freud.locality.LinkCell` |
+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+
| exclude_ii     | Whether or not to include neighbors with the same index in the array  | bool      | True/False                | :class:`freud.locality.AABBQuery`, :class:`freud.locality.LinkCell` |
+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+
| r_guess        | Initial search distance for sequence of ball queries                  | float     | r_guess > 0               | :class:`freud.locality.AABBQuery`                                   |
+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+
| scale          | Scale factor for r_guess when not enough neighbors are found          | float     | scale > 1                 | :class:`freud.locality.AABBQuery`                                   |
+----------------+-----------------------------------------------------------------------+-----------+---------------------------+---------------------------------------------------------------------+

Query Modes
===========

Ball Query (Distance Cutoff)
----------------------------

A ball query finds all particles within a specified radial distance of the provided query points.
This query is executed when ``mode='ball'``.
As described in the table above, this mode can be coupled with filters for a minimum distance (``r_min``) and/or self-exclusion (``exclude_ii``).

Nearest Neighbors Query (Fixed Number of Neighbors)
---------------------------------------------------

A nearest neighbor query (sometimes called :math:`k`-nearest neighbors) finds a desired number of neighbor points for each query point, ordered by distance to the query point.
This query is executed when ``mode='nearest'``.
As described in the table above, this mode can be coupled with filters for a maximum distance (``r_max``), minimum distance (``r_min``), and/or self-exclusion (``exclude_ii``).

Mode Deduction
--------------

The ``mode`` query argument specifies the type of query that is being performed, and it therefore governs how other arguments are interpreted.
In most cases, however, the query mode can be deduced from the set of query arguments.
Specifically, any query with the ``num_neighbors`` key set is assumed to be a query with ``mode='nearest'``.
One of ``num_neighbors`` or ``r_max`` must always be specified to form a valid set of query arguments.
Specifying the ``mode`` key explicitly will ensure that querying behavior is consistent if additional query modes are added to **freud**.


Query Results
=============

Although they don't typically need to be operated on directly, it can be useful to know a little about the objects returned by queries.
The :class:`freud.locality.NeighborQueryResult` stores the ``query_points`` passed to a ``query`` and returns neighbors for them one at a time (like any Python :class:`iterator`).
The primary goal of the result class is to support easy iteration and conversion to more persistent formats.
Since it is an iterator, you can use any typical Python approach to consuming it, including passing it to :class:`list` to build a list of the neighbors.
For a more **freud**-friendly approach, you can use the :meth:`toNeighborList <freud.locality.NeighborQueryResult.toNeighborList>` method to convert the object into a **freud** :class:`freud.locality.NeighborList`.
Under the hood, the underlying C++ classes loop through candidate points and identifying neighbors for each ``query_point``; this is the same process that occurs when ``Compute classes`` employ :class:`NeighborQuery <freud.locality.NeighborQuery>` objects for finding neighbors on-the-fly, but in that case it all happens on the C++ side.


Custom NeighborLists
====================

Thus far, we've mostly discussed :class:`NeighborLists <freud.locality.NeighborList>` as a way to persist neighbor information beyond a single query.
In :ref:`optimizing`, more guidance is provided on how you can use these objects to speed up certain uses of **freud**.
However, these objects are also extremely useful because they provide a *completely customizable* way to specify neighbors to **freud**.
Of particular note here is the :meth:`freud.locality.NeighborList.from_arrays` factory function that allows you to make :class:`freud.locality.NeighborList` objects by directly specifying the ``(i, j)`` pairs that should be in the list.
This kind of explicit construction of the list enables custom analyses that would otherwise be impossible.
For example, consider a molecular dynamics simulation in which particles only interact via extremely short-ranged patches on their surface, and that particles should only be considered bonded if their patches are actually interacting, irrespective of how close together the particles themselves are.
This type of neighbor interaction cannot be captured by any normal querying mode, but could be constructed by the user and then fed to **freud** for downstream analysis.

Nearest Neighbor Asymmetry
==========================

There is one important but easily overlooked detail associated with using query arguments with mode ``'nearest'``.
Consider a simple example of three points on the x-axis located at -1, 0, and 2 (and assume the box is of dimensions :math:`(100, 100, 100)`, i.e. sufficiently large that periodicity plays no role):

.. code-block:: python

    box = [100, 100, 100]
    points = [[-1, 0, 0], [0, 0, 0], [2, 0, 0]]
    query_args = dict(mode='nearest', num_neighbors=1, exclude_ii=True)
    list(freud.locality.AABBQuery(box, points).query(points, query_args))
    # Output: [(0, 1, 1), (1, 0, 1), (2, 1, 2)]

Evidently, the calculation is not symmetric.
This feature of nearest neighbor queries can have unexpected side effects if a ``PairCompute`` is performed using distinct ``points`` and ``query_points`` and the two are interchanged.
In such cases, users should always keep in mind that **freud** promises that every ``query_point`` will end up with ``num_neighbors`` points (assuming no hard cutoff ``r_max`` is imposed and enough points are present in the system).
However, it is possible (and indeed likely) that any given ``point`` will have more or fewer than that many neighbors.
This distinction can be particularly tricky for calculations that depend on vector directionality: **freud** imposes the convention that bond vectors always point from ``query_point`` to ``point``, so users of calculations like PMFTs where directionality is important should keep this in mind.
