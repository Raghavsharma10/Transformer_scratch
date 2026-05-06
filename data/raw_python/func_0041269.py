def p_Dictionary(p):
  """Dictionary : dictionary IDENTIFIER Inheritance "{" DictionaryMembers "}" ";"
  """
  p[0] = model.Dictionary(name=p[2], parent=p[3], members=p[5])