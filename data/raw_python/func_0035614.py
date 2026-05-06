def _parse_fmt(fmt, color_key='colors', ls_key='linestyles',
               marker_key='marker'):
  '''Modified from matplotlib's _process_plot_format function.'''
  try:  # Is fmt just a colorspec?
    color = mcolors.colorConverter.to_rgb(fmt)
  except ValueError:
    pass  # No, not just a color.
  else:
    # Either a color or a numeric marker style
    if fmt not in mlines.lineMarkers:
      return {color_key:color}

  result = dict()
  # handle the multi char special cases and strip them from the string
  if fmt.find('--') >= 0:
    result[ls_key] = '--'
    fmt = fmt.replace('--', '')
  if fmt.find('-.') >= 0:
    result[ls_key] = '-.'
    fmt = fmt.replace('-.', '')
  if fmt.find(' ') >= 0:
    result[ls_key] = 'None'
    fmt = fmt.replace(' ', '')

  for c in list(fmt):
    if c in mlines.lineStyles:
      if ls_key in result:
        raise ValueError('Illegal format string; two linestyle symbols')
      result[ls_key] = c
    elif c in mlines.lineMarkers:
      if marker_key in result:
        raise ValueError('Illegal format string; two marker symbols')
      result[marker_key] = c
    elif c in mcolors.colorConverter.colors:
      if color_key in result:
        raise ValueError('Illegal format string; two color symbols')
      result[color_key] = c
    else:
      raise ValueError('Unrecognized character %c in format string' % c)
  return result