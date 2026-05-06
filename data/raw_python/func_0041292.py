def p_ExtendedAttributeArgList(p):
  """ExtendedAttributeArgList : IDENTIFIER "(" ArgumentList ")"
  """
  p[0] = model.ExtendedAttribute(
    value=model.ExtendedAttributeValue(name=p[1], arguments=p[3]))