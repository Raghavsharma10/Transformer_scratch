def p_Const(p):
  """Const : const ConstType IDENTIFIER "=" ConstValue ";"
  """
  p[0] = model.Const(type=p[2], name=p[3], value=p[5])