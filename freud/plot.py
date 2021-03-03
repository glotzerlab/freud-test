# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import freud
import io
import numpy as np
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    raise ImportError('matplotlib must be installed for freud.plot.')


def _ax_to_bytes(ax):
    """Helper function to convert figure to png file.

    Args:
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.

    Returns:
        bytes: Byte representation of the diagram in png format.
    """
    f = io.BytesIO()
    # Sets an Agg backend so this figure can be rendered
    fig = ax.figure
    FigureCanvasAgg(fig)
    fig.savefig(f, format='png')
    fig.clf()
    return f.getvalue()


def _set_3d_axes_equal(ax, limits=None):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc. This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
        ax (:class:`matplotlib.axes.Axes`): Axes object.
        limits (:math:`(3, 2)` :class:`np.ndarray`):
            Axis limits in the form
            :code:`[[xmin, xmax], [ymin, ymax], [zmin, zmax]]`. If
            :code:`None`, the limits are auto-detected (Default value =
            :code:`None`).
    """
    # Adapted from https://stackoverflow.com/a/50664367

    if limits is None:
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    else:
        limits = np.asarray(limits)
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
    return ax


def box_plot(box, title=None, ax=None, image=[0, 0, 0], *args, **kwargs):
    """Helper function to plot a :class:`~.box.Box` object.

    Args:
        box (:class:`~.box.Box`):
            Simulation box.
        title (str):
            Title of the graph. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            If plotting a 3D box, the axes must be 3D.
            (Default value = :code:`None`).
        image (list):
            The periodic image location at which to draw the box (Default
            value = :code:`[0, 0, 0]`).
        ``*args``, ``**kwargs``:
            All other arguments are passed on to
            :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
            :meth:`matplotlib.axes.Axes.plot`.
    """
    box = freud.box.Box.from_box(box)

    if ax is None:
        fig = plt.figure()
        if box.is2D:
            ax = fig.subplots()
        else:
            # This import registers the 3d projection
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = fig.add_subplot(111, projection='3d')

    if box.is2D:
        # Draw 2D box
        corners = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
        # Need to copy the last point so that the box is closed.
        corners.append(corners[0])
        corners = np.asarray(corners)
        corners += np.asarray(image)
        corners = box.make_absolute(corners)[:, :2]
        color = kwargs.pop('color', 'k')
        ax.plot(corners[:, 0], corners[:, 1], color=color, *args, **kwargs)
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
    else:
        # Draw 3D box
        corners = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        corners += np.asarray(image)
        corners = box.make_absolute(corners)
        paths = [corners[[0, 1, 3, 2, 0]],
                 corners[[4, 5, 7, 6, 4]],
                 corners[[0, 4]], corners[[1, 5]],
                 corners[[2, 6]], corners[[3, 7]]]
        for path in paths:
            color = kwargs.pop('color', 'k')
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        limits = [[corners[0, 0], corners[-1, 0]],
                  [corners[0, 1], corners[-1, 1]],
                  [corners[0, 2], corners[-1, 2]]]
        _set_3d_axes_equal(ax, limits)

    return ax


def system_plot(system, title=None, ax=None, *args, **kwargs):
    """Helper function to plot a system object.

    Args:
        system
            Any object that is a valid argument to
            :class:`freud.locality.NeighborQuery.from_system`.
        title (str):
            Title of the plot. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).
    """
    system = freud.locality.NeighborQuery.from_system(system)

    if ax is None:
        fig = plt.figure()
        if system.box.is2D:
            ax = fig.subplots()
        else:
            # This import registers the 3d projection
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = fig.add_subplot(111, projection='3d')

    if system.box.is2D:
        box_plot(system.box, ax=ax)
        sc = ax.scatter(system.points[:, 0],
                        system.points[:, 1],
                        *args, **kwargs)
        ax.set_aspect('equal', 'datalim')
    else:
        box_plot(system.box, ax=ax)
        sc = ax.scatter(system.points[:, 0],
                        system.points[:, 1],
                        system.points[:, 2],
                        *args, **kwargs)
        box_min = system.box.make_absolute([0, 0, 0])
        box_max = system.box.make_absolute([1, 1, 1])
        points_min = np.min(system.points, axis=0)
        points_max = np.max(system.points, axis=0)
        limits = [[np.min([box_min[i], points_min[i]]),
                   np.max([box_max[i], points_max[i]])] for i in range(3)]
        _set_3d_axes_equal(ax, limits=limits)

    return ax, sc


def bar_plot(x, height, title=None, xlabel=None, ylabel=None, ax=None):
    """Helper function to draw a bar graph.

    Args:
        x (list): x values of the bar graph.
        height (list): Height values corresponding to :code:`x`.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    ax.bar(x=x, height=height)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    return ax


def clusters_plot(keys, freqs, num_clusters_to_plot=10, ax=None):
    """Helper function to plot most frequent clusters in a bar graph.

    Args:
        keys (list): Cluster keys.
        freqs (list): Number of particles in each clusters.
        num_clusters_to_plot (unsigned int): Number of largest clusters to
            plot.
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    count_sorted = sorted([(freq, key)
                          for key, freq in zip(keys, freqs)],
                          key=lambda x: -x[0])
    sorted_freqs = [i[0] for i in count_sorted[:num_clusters_to_plot]]
    sorted_keys = [str(i[1]) for i in count_sorted[:num_clusters_to_plot]]
    return bar_plot(sorted_keys, sorted_freqs, title="Cluster Frequency",
                    xlabel="Keys of {} largest clusters (total clusters: "
                           "{})".format(len(sorted_freqs), len(freqs)),
                    ylabel="Number of particles", ax=ax)


def line_plot(x, y, title=None, xlabel=None, ylabel=None, ax=None):
    """Helper function to draw a line graph.

    Args:
        x (list): x values of the line graph.
        y (list): y values corresponding to :code:`x`.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def histogram_plot(values, title=None, xlabel=None, ylabel=None, ax=None):
    """Helper function to draw a histogram graph.

    Args:
        values (list): values of the histogram.
        title (str): Title of the graph. (Default value = :code:`None`).
        xlabel (str): Label of x axis. (Default value = :code:`None`).
        ylabel (str): Label of y axis. (Default value = :code:`None`).
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    ax.hist(values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def pmft_plot(pmft, ax=None):
    """Helper function to draw 2D PMFT diagram.

    Args:
        pmft (:class:`freud.pmft.PMFTXY2D`):
            PMFTXY2D instance.
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from matplotlib.colorbar import Colorbar

    # Plot figures
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    pmft_arr = np.copy(pmft.PMFT)
    pmft_arr[np.isinf(pmft_arr)] = np.nan

    xlims = (pmft.X[0], pmft.X[-1])
    ylims = (pmft.Y[0], pmft.Y[-1])
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.xaxis.set_ticks([i for i in range(int(xlims[0]), int(xlims[1]+1))])
    ax.yaxis.set_ticks([i for i in range(int(ylims[0]), int(ylims[1]+1))])
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('PMFT')

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="10%")

    im = ax.imshow(np.flipud(pmft_arr),
                   extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                   interpolation='nearest', cmap='viridis',
                   vmin=-2.5, vmax=3.0)

    cb = Colorbar(cax, im)
    cb.set_label(r"$k_B T$")

    return ax


def density_plot(density, box, ax=None):
    R"""Helper function to plot density diagram.

    Args:
        density (:math:`\left(N_x, N_y\right)` :class:`numpy.ndarray`):
            Array containing density.
        box (:class:`freud.box.Box`):
            Simulation box.
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from matplotlib.colorbar import Colorbar

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    xlims = (-box.Lx/2, box.Lx/2)
    ylims = (-box.Ly/2, box.Ly/2)

    ax.set_title('Gaussian Density')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="10%")

    im = ax.imshow(np.flipud(density.T),
                   extent=[xlims[0], xlims[1], ylims[0], ylims[1]])

    cb = Colorbar(cax, im)
    cb.set_label("Density")

    return ax


def voronoi_plot(box, polytopes, ax=None, color_by_sides=True, cmap=None):
    """Helper function to draw 2D Voronoi diagram.

    Args:
        box (:class:`freud.box.Box`):
            Simulation box.
        polytopes (:class:`numpy.ndarray`):
            Array containing Voronoi polytope vertices.
        ax (:class:`matplotlib.axes.Axes`): Axes object to plot.
            If :code:`None`, make a new axes and figure object.
            (Default value = :code:`None`).
        color_by_sides (bool):
            If :code:`True`, color cells by the number of sides.
            If :code:`False`, random colors are used for each cell.
            (Default value = :code:`True`).
        cmap (str):
            Colormap name to use (Default value = :code:`None`).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    from matplotlib import cm
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from matplotlib.colorbar import Colorbar

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    # Draw Voronoi polytopes
    patches = [Polygon(poly[:, :2]) for poly in polytopes]
    patch_collection = PatchCollection(patches, edgecolors='black', alpha=0.4)

    if color_by_sides:
        colors = np.array([len(poly) for poly in polytopes])
        num_colors = np.ptp(colors) + 1
    else:
        colors = np.random.RandomState().permutation(np.arange(len(patches)))
        num_colors = np.unique(colors).size

    # Ensure we have enough colors to uniquely identify the cells
    if cmap is None:
        if color_by_sides and num_colors <= 10:
            cmap = 'tab10'
        else:
            if num_colors > 20:
                warnings.warn('More than 20 unique colors were requested. '
                              'Consider providing a colormap to the cmap '
                              'argument.', UserWarning)
            cmap = 'tab20'
    cmap = cm.get_cmap(cmap, num_colors)
    bounds = np.arange(np.min(colors), np.max(colors)+1)

    patch_collection.set_array(np.array(colors)-0.5)
    patch_collection.set_cmap(cmap)
    patch_collection.set_clim(bounds[0]-0.5, bounds[-1]+0.5)
    ax.add_collection(patch_collection)

    # Draw box
    corners = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]
    # Need to copy the last point so that the box is closed.
    corners.append(corners[0])
    corners = box.make_absolute(corners)[:, :2]
    ax.plot(corners[:, 0], corners[:, 1], color='k')

    # Set title, limits, aspect
    ax.set_title('Voronoi Diagram')
    ax.set_xlim((np.min(corners[:, 0]), np.max(corners[:, 0])))
    ax.set_ylim((np.min(corners[:, 1]), np.max(corners[:, 1])))
    ax.set_aspect('equal', 'datalim')

    # Add colorbar for number of sides
    if color_by_sides:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="10%")
        cb = Colorbar(cax, patch_collection)
        cb.set_label("Number of sides")
        cb.set_ticks(bounds)
    return ax


def diffraction_plot(diffraction, k_values, ax=None, cmap='afmhot',
                     vmin=4e-6, vmax=0.7):
    """Helper function to plot diffraction pattern.

    Args:
        diffraction (:class:`numpy.ndarray`):
            Diffraction image data.
        k_values (:class:`numpy.ndarray`):
            :math:`k` value magnitudes for each bin of the diffraction image.
        ax (:class:`matplotlib.axes.Axes`):
            Axes object to plot. If :code:`None`, make a new axes and figure
            object (Default value = :code:`None`).
        cmap (str):
            Colormap name to use (Default value = :code:`'afmhot'`).
        vmin (float):
            Minimum of the color scale (Default value = 4e-6).
        vmax (float):
            Maximum of the color scale (Default value = 0.7).

    Returns:
        :class:`matplotlib.axes.Axes`: Axes object with the diagram.
    """
    import matplotlib.colors
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from matplotlib.colorbar import Colorbar

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    # Plot the diffraction image and color bar
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(np.clip(diffraction, vmin, vmax),
                   interpolation='nearest', cmap=cmap, norm=norm)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="10%")
    cb = Colorbar(cax, im)
    cb.set_label(r"$S(\vec{k})$")

    # Determine the number of ticks on the axis
    grid_size = diffraction.shape[0]
    num_ticks = len([i for i in ax.xaxis.get_ticklocs()
                     if 0 <= i <= grid_size])

    # Ensure there are an odd number of ticks, so that there is a tick at zero
    num_ticks += (1 - num_ticks % 2)
    ticks = np.linspace(0, grid_size, num_ticks)

    # Set tick locations and labels
    tick_labels = np.interp(ticks, range(grid_size), k_values)
    tick_labels = ['{:.3g}'.format(x) for x in tick_labels]
    ax.xaxis.set_ticks(ticks)
    ax.xaxis.set_ticklabels(tick_labels)
    ax.yaxis.set_ticks(ticks)
    ax.yaxis.set_ticklabels(tick_labels)

    # Set title, limits, aspect
    ax.set_title('Diffraction Pattern')
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')

    return ax
