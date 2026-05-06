def _layer_to_path_gen(layer):
    """Generates an SVG path from a given layer.

    Args:
        layer (layer): the layer to convert

    Yields:
        str: the next component of the path
    """
    draw = False
    for x, y in zip(*layer):
        if np.isnan(x) or np.isnan(y):
            draw = False
        elif not draw:
            yield 'M {} {}'.format(x, y)
            draw = True
        else:
            yield 'L {} {}'.format(x, y)