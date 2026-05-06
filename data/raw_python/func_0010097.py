def _websafe_component(c, alt=False):
  """Convert a color component to its web safe equivalent.

  Parameters:
    :c:
      The component value [0...1]
    :alt:
      If True, return the alternative value instead of the nearest one.

  Returns:
    The web safe equivalent of the component value.

  """
  # This sucks, but floating point between 0 and 1 is quite fuzzy...
  # So we just change the scale a while to make the equality tests
  # work, otherwise it gets wrong at some decimal far to the right.
  sc = c * 100.0

  # If the color is already safe, return it straight away
  d = sc % 20
  if d==0: return c

  # Get the lower and upper safe values
  l = sc - d
  u = l + 20

  # Return the 'closest' value according to the alt flag
  if alt:
    if (sc-l) >= (u-sc): return l/100.0
    else: return u/100.0
  else:
    if (sc-l) >= (u-sc): return u/100.0
    else: return l/100.0