def p_NonAnyType_object(p):
  """NonAnyType : object TypeSuffix"""
  p[0] = helper.unwrapTypeSuffix(model.SimpleType(
    type=model.SimpleType.OBJECT), p[2])