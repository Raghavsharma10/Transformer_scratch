def get_frag_shader(volumes, clipped=False, n_volume_max=5):
    """
    Get the fragment shader code - we use the shader_program object to determine
    which layers are enabled and therefore what to include in the shader code.
    """

    declarations = ""
    before_loop = ""
    in_loop = ""
    after_loop = ""

    for index in range(n_volume_max):
        declarations += "uniform $sampler_type u_volumetex_{0:d};\n".format(index)
        before_loop += "dummy = $sample(u_volumetex_{0:d}, loc).g;\n".format(index)

    declarations += "uniform $sampler_type dummy1;\n"
    declarations += "float dummy;\n"

    for label in sorted(volumes):

        index = volumes[label]['index']

        # Global declarations
        declarations += "uniform float u_weight_{0:d};\n".format(index)
        declarations += "uniform int u_enabled_{0:d};\n".format(index)

        # Declarations before the raytracing loop
        before_loop += "float max_val_{0:d} = 0;\n".format(index)

        # Calculation inside the main raytracing loop

        in_loop += "if(u_enabled_{0:d} == 1) {{\n\n".format(index)

        if clipped:
            in_loop += ("if(loc.r > u_clip_min.r && loc.r < u_clip_max.r &&\n"
                        "   loc.g > u_clip_min.g && loc.g < u_clip_max.g &&\n"
                        "   loc.b > u_clip_min.b && loc.b < u_clip_max.b) {\n\n")

        in_loop += "// Sample texture for layer {0}\n".format(label)
        in_loop += "val = $sample(u_volumetex_{0:d}, loc).g;\n".format(index)

        if volumes[label].get('multiply') is not None:
            index_other = volumes[volumes[label]['multiply']]['index']
            in_loop += ("if (val != 0) {{ val *= $sample(u_volumetex_{0:d}, loc).g; }}\n"
                        .format(index_other))

        in_loop += "max_val_{0:d} = max(val, max_val_{0:d});\n\n".format(index)

        if clipped:
            in_loop += "}\n\n"

        in_loop += "}\n\n"

        # Calculation after the main loop

        after_loop += "// Compute final color for layer {0}\n".format(label)
        after_loop += ("color = $cmap{0:d}(max_val_{0:d});\n"
                       "color.a *= u_weight_{0:d};\n"
                       "total_color += color.a * color;\n"
                       "max_alpha = max(color.a, max_alpha);\n"
                       "count += color.a;\n\n").format(index)

    if not clipped:
        before_loop += "\nfloat val3 = u_clip_min.g + u_clip_max.g;\n\n"

    # Code esthetics
    before_loop = indent(before_loop, " " * 4).strip()
    in_loop = indent(in_loop, " " * 16).strip()
    after_loop = indent(after_loop, " " * 4).strip()

    return FRAG_SHADER.format(declarations=declarations,
                              before_loop=before_loop,
                              in_loop=in_loop,
                              after_loop=after_loop)