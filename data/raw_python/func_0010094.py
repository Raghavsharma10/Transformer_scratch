def html_to_rgb(html):
  """Convert the HTML color to (r, g, b).

  Parameters:
    :html:
      the HTML definition of the color (#RRGGBB or #RGB or a color name).

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  Throws:
    :ValueError:
      If html is neither a known color name or a hexadecimal RGB
      representation.

  >>> '(%g, %g, %g)' % html_to_rgb('#ff8000')
  '(1, 0.501961, 0)'
  >>> '(%g, %g, %g)' % html_to_rgb('ff8000')
  '(1, 0.501961, 0)'
  >>> '(%g, %g, %g)' % html_to_rgb('#f60')
  '(1, 0.4, 0)'
  >>> '(%g, %g, %g)' % html_to_rgb('f60')
  '(1, 0.4, 0)'
  >>> '(%g, %g, %g)' % html_to_rgb('lemonchiffon')
  '(1, 0.980392, 0.803922)'

  """
  html = html.strip().lower()
  if html[0]=='#':
    html = html[1:]
  elif html in NAMED_COLOR:
    html = NAMED_COLOR[html][1:]

  if len(html)==6:
    rgb = html[:2], html[2:4], html[4:]
  elif len(html)==3:
    rgb = ['%c%c' % (v,v) for v in html]
  else:
    raise ValueError("input #%s is not in #RRGGBB format" % html)

  return tuple(((int(n, 16) / 255.0) for n in rgb))