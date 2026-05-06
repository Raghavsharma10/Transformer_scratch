def p_ExtendedAttributeNamedArgList(p):
  """ExtendedAttributeNamedArgList : IDENTIFIER "=" IDENTIFIER "(" ArgumentList ")"
  """
  p[0] = model.ExtendedAttribute(
    name=p[1],
    value=model.ExtendedAttributeValue(name=p[3], arguments=p[5]))