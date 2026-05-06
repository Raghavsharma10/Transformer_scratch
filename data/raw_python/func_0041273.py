def p_CallbackRest(p):
  """CallbackRest : IDENTIFIER "=" ReturnType "(" ArgumentList ")" ";"
  """
  p[0] = model.Callback(name=p[1], return_type=p[3], arguments=p[5])