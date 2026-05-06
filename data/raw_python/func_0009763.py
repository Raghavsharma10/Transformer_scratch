def plot_to_svg(plot, width, height, unit=''):
    """Converts a plot (list of layers) into an SVG document.

    Args:
        plot (list): list of layers that make up the plot
        width (float): the width of the resulting image
        height (float): the height of the resulting image
        unit (str): the units of the resulting image if not pixels

    Returns:
        str: A stringified XML document representing the image
    """
    flipped_plot = [(x, -y) for x, y in plot]
    aspect_ratio = height / width
    view_box = calculate_view_box(flipped_plot, aspect_ratio=aspect_ratio)
    view_box_str = '{} {} {} {}'.format(*view_box)
    stroke_thickness = STROKE_THICKNESS * (view_box[2])

    svg = ET.Element('svg', attrib={
        'xmlns': 'http://www.w3.org/2000/svg',
        'xmlns:inkscape': 'http://www.inkscape.org/namespaces/inkscape',
        'width': '{}{}'.format(width, unit),
        'height': '{}{}'.format(height, unit),
        'viewBox': view_box_str})

    for i, layer in enumerate(flipped_plot):
        group = ET.SubElement(svg, 'g', attrib={
            'inkscape:label': '{}-layer'.format(i),
            'inkscape:groupmode': 'layer',
        })

        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        ET.SubElement(group, 'path', attrib={
            'style': 'stroke-width: {}; stroke: {};'.format(stroke_thickness, color),
            'fill': 'none',
            'd': layer_to_path(layer)
        })

    try:
        return ET.tostring(svg, encoding='unicode')
    except LookupError:
        # Python 2.x
        return ET.tostring(svg)