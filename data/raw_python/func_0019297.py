def _assignrepr_bracketed2(assignrepr_bracketed1, values, prefix, width=None):
    """Return a prefixed, wrapped and properly aligned bracketed string
    representation of the given 2-dimensional value matrix using function
    |repr|."""
    brackets = getattr(assignrepr_bracketed1, '_brackets')
    prefix += brackets[0]
    lines = []
    blanks = ' '*len(prefix)
    for (idx, subvalues) in enumerate(values):
        if idx == 0:
            lines.append(assignrepr_bracketed1(subvalues, prefix, width))
        else:
            lines.append(assignrepr_bracketed1(subvalues, blanks, width))
        lines[-1] += ','
    if (len(values) > 1) or (brackets != '()'):
        lines[-1] = lines[-1][:-1]
    lines[-1] += brackets[1]
    return '\n'.join(lines)