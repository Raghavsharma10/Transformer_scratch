def p_ConstValue_float(p):
  """ConstValue : FLOAT"""
  p[0] = model.Value(type=model.Value.FLOAT, value=p[1])