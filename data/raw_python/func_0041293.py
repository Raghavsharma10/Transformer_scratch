def p_ExtendedAttributeIdent(p):
  """ExtendedAttributeIdent : IDENTIFIER "=" IDENTIFIER"""
  p[0] = model.ExtendedAttribute(
    name=p[1],
    value=model.ExtendedAttributeValue(name=p[3]))