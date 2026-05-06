def p_DefaultValue_string(p):
  """DefaultValue : STRING"""
  p[0] = model.Value(type=model.Value.STRING, value=p[1])