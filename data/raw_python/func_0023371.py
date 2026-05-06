def _glsl_mix(controls=None):
    """Generate a GLSL template function from a given interpolation patterns
    and control points."""
    assert (controls[0], controls[-1]) == (0., 1.)
    ncolors = len(controls)
    assert ncolors >= 2
    if ncolors == 2:
        s = "    return mix($color_0, $color_1, t);\n"
    else:
        s = ""
        for i in range(ncolors-1):
            if i == 0:
                ifs = 'if (t < %.6f)' % (controls[i+1])
            elif i == (ncolors-2):
                ifs = 'else'
            else:
                ifs = 'else if (t < %.6f)' % (controls[i+1])
            adj_t = '(t - %s) / %s' % (controls[i],
                                       controls[i+1] - controls[i])
            s += ("%s {\n    return mix($color_%d, $color_%d, %s);\n} " %
                  (ifs, i, i+1, adj_t))
    return "vec4 colormap(float t) {\n%s\n}" % s