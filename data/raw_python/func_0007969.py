def plot2d(points, cells, mesh_color="k", show_axes=False):
    """Plot a 2D mesh using matplotlib.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure()
    ax = fig.gca()
    plt.axis("equal")
    if not show_axes:
        ax.set_axis_off()

    xmin = numpy.amin(points[:, 0])
    xmax = numpy.amax(points[:, 0])
    ymin = numpy.amin(points[:, 1])
    ymax = numpy.amax(points[:, 1])

    width = xmax - xmin
    xmin -= 0.1 * width
    xmax += 0.1 * width

    height = ymax - ymin
    ymin -= 0.1 * height
    ymax += 0.1 * height

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    edge_nodes, _ = create_edges(cells)

    # Get edges, cut off z-component.
    e = points[edge_nodes][:, :, :2]
    line_segments = LineCollection(e, color=mesh_color)
    ax.add_collection(line_segments)
    return fig