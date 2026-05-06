def convert_shaders(convert, shaders):
    """ Modify shading code so that we can write code once
    and make it run "everywhere".
    """

    # New version of the shaders
    out = []

    if convert == 'es2':

        for isfragment, shader in enumerate(shaders):
            has_version = False
            has_prec_float = False
            has_prec_int = False
            lines = []
            # Iterate over lines
            for line in shader.lstrip().splitlines():
                if line.startswith('#version'):
                    has_version = True
                    continue
                if line.startswith('precision '):
                    has_prec_float = has_prec_float or 'float' in line
                    has_prec_int = has_prec_int or 'int' in line
                lines.append(line.rstrip())
            # Write
            # BUG: fails on WebGL (Chrome)
            # if True:
            #     lines.insert(has_version, '#line 0')
            if not has_prec_float:
                lines.insert(has_version, 'precision highp float;')
            if not has_prec_int:
                lines.insert(has_version, 'precision highp int;')
            # BUG: fails on WebGL (Chrome)
            # if not has_version:
            #     lines.insert(has_version, '#version 100')
            out.append('\n'.join(lines))

    elif convert == 'desktop':

        for isfragment, shader in enumerate(shaders):
            has_version = False
            lines = []
            # Iterate over lines
            for line in shader.lstrip().splitlines():
                has_version = has_version or line.startswith('#version')
                if line.startswith('precision '):
                    line = ''
                for prec in (' highp ', ' mediump ', ' lowp '):
                    line = line.replace(prec, ' ')
                lines.append(line.rstrip())
            # Write
            if not has_version:
                lines.insert(0, '#version 120\n')
            out.append('\n'.join(lines))

    else:
        raise ValueError('Cannot convert shaders to %r.' % convert)

    return tuple(out)