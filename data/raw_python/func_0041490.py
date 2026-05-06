def t_IDENTIFIER(t):
  r"[A-Z_a-z][0-9A-Z_a-z]*"
  if t.value in keywords:
    t.type = t.value
  return t