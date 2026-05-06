def rgb_to_html(r, g=None, b=None):
  """Convert the color from (r, g, b) to #RRGGBB.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    A CSS string representation of this color (#RRGGBB).

  >>> rgb_to_html(1, 0.5, 0)
  '#ff8000'

  """
  if type(r) in [list,tuple]:
    r, g, b = r
  return '#%02x%02x%02x' % tuple((min(round(v*255), 255) for v in (r, g, b)))