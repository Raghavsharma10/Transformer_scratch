def p_UnionType(p):
  """UnionType : "(" UnionMemberType or UnionMemberType UnionMemberTypes ")"
  """
  t = [p[2]] + [p[4]] + p[5]
  p[0] = model.UnionType(t=t)