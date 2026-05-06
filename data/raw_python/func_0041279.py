def p_ConstValue_null(p):
  """ConstValue : null"""
  p[0] = model.Value(type=model.Value.NULL, value=p[1])