def prepare_axes(axes, title, size, cmap=None):
    """Prepares an axes object for clean plotting.

    Removes x and y axes labels and ticks, sets the aspect ratio to be
    equal, uses the size to determine the drawing area and fills the image
    with random colors as visual feedback.

    Creates an AxesImage to be shown inside the axes object and sets the
    needed properties.

    Args:
        axes:  The axes object to modify.
        title: The title.
        size:  The size of the expected image.
        cmap:  The colormap if a custom color map is needed.
                (Default: None)
    Returns:
        The AxesImage's handle.
    """
    if axes is None:
        return None

    # prepare axis itself
    axes.set_xlim([0, size[1]])
    axes.set_ylim([size[0], 0])
    axes.set_aspect('equal')

    axes.axis('off')
    if isinstance(cmap, str):
        title = '{} (cmap: {})'.format(title, cmap)
    axes.set_title(title)

    # prepare image data
    axes_image = image.AxesImage(axes, cmap=cmap,
                                 extent=(0, size[1], size[0], 0))
    axes_image.set_data(np.random.random((size[0], size[1], 3)))

    axes.add_image(axes_image)
    return axes_image