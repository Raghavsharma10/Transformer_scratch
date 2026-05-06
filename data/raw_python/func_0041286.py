def p_UnionMemberType_anyType(p):
  """UnionMemberType : any "[" "]" TypeSuffix"""
  p[0] = helper.unwrapTypeSuffix(model.Array(t=model.SimpleType(
    type=model.SimpleType.ANY)), p[4])