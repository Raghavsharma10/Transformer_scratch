def p_ConstValue_integer(p):
  """ConstValue : INTEGER"""
  p[0] = model.Value(type=model.Value.INTEGER, value=p[1])