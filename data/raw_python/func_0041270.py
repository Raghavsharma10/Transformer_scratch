def p_DictionaryMember(p):
  """DictionaryMember : Type IDENTIFIER Default ";"
  """
  p[0] = model.DictionaryMember(type=p[1], name=p[2], default=p[3])