def p_SingleType_any(p):
  """SingleType : any TypeSuffixStartingWithArray"""
  p[0] = helper.unwrapTypeSuffix(model.SimpleType(
    model.SimpleType.ANY), p[2])