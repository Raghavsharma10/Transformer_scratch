def p_NonAnyType_interface(p):
  """NonAnyType : IDENTIFIER TypeSuffix"""
  p[0] = helper.unwrapTypeSuffix(model.InterfaceType(name=p[1]), p[2])