def _process_glsl_template(template, colors):
    """Replace $color_i by color #i in the GLSL template."""
    for i in range(len(colors) - 1, -1, -1):
        color = colors[i]
        assert len(color) == 4
        vec4_color = 'vec4(%.3f, %.3f, %.3f, %.3f)' % tuple(color)
        template = template.replace('$color_%d' % i, vec4_color)
    return template