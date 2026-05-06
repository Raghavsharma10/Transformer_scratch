def p_NonAnyType_domString(p):
  """NonAnyType : DOMString TypeSuffix"""
  p[0] = helper.unwrapTypeSuffix(model.SimpleType(
    type=model.SimpleType.DOMSTRING), p[2])