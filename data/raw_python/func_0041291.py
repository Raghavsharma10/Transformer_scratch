def p_ExtendedAttributeNoArgs(p):
  """ExtendedAttributeNoArgs : IDENTIFIER"""
  p[0] = model.ExtendedAttribute(
    value=model.ExtendedAttributeValue(name=p[1]))