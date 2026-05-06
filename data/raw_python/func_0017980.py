def unquote(s):
  """Unquote the indicated string."""
  # Ignore the left- and rightmost chars (which should be quotes).
  # Use the Python engine to decode the escape sequence
  i, N = 1, len(s) - 1
  ret = []
  while i < N:
    if s[i] == '\\' and i < N - 1:
      ret.append(UNQUOTE_MAP.get(s[i+1], s[i+1]))
      i += 2
    else:
      ret.append(s[i])
      i += 1
  return ''.join(ret)