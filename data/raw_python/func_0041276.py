def p_ConstValue_boolean(p):
  """ConstValue : BooleanLiteral"""
  p[0] = model.Value(type=model.Value.BOOLEAN, value=p[1])