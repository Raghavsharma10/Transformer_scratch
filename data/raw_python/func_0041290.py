def p_NonAnyType(p):
  """NonAnyType : Date TypeSuffix"""
  p[0] = helper.unwrapTypeSuffix(model.SimpleType(
    type=model.SimpleType.DATE), p[2])