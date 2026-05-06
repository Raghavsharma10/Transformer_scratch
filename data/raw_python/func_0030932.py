def parseColors(colors, defaultColor):
    """
    Parse command line color information.

    @param colors: A C{list} of space separated "value color" strings, such as
        ["0.9 red", "0.75 rgb(23, 190, 207)", "0.1 #CF3CF3"].
    @param defaultColor: The C{str} color to use for cells that do not reach
        the identity fraction threshold of any color in C{colors}.
    @return: A C{list} of (threshold, color) tuples, where threshold is a
        C{float} (from C{colors}) and color is a C{str} (from C{colors}). The
        list will be sorted by decreasing threshold values.
    """
    result = []
    if colors:
        for colorInfo in colors:
            fields = colorInfo.split(maxsplit=1)
            if len(fields) == 2:
                threshold, color = fields
                try:
                    threshold = float(threshold)
                except ValueError:
                    print('--color arguments must be given as space-separated '
                          'pairs of "value color" where the value is a '
                          'numeric identity threshold. Your value %r is not '
                          'numeric.' % threshold, file=sys.stderr)
                    sys.exit(1)
                if 0.0 > threshold > 1.0:
                    print('--color arguments must be given as space-separated '
                          'pairs of "value color" where the value is a '
                          'numeric identity threshold from 0.0 to 1.0. Your '
                          'value %r is not in that range.' % threshold,
                          file=sys.stderr)
                    sys.exit(1)

                result.append((threshold, color))
            else:
                print('--color arguments must be given as space-separated '
                      'pairs of "value color". You have given %r, which does '
                      'not contain a space.' % colorInfo, file=sys.stderr)
                sys.exit(1)

    result.sort(key=itemgetter(0), reverse=True)

    if not result or result[-1][0] > 0.0:
        result.append((0.0, defaultColor))

    return result