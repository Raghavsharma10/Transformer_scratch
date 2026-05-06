def p_Attribute(p):
  """Attribute : Inherit ReadOnly attribute Type IDENTIFIER ";"
  """
  p[0] = model.Attribute(inherit=p[1], readonly=p[2], type=p[4], name=p[5])