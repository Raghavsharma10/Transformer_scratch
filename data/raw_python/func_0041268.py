def p_Interface(p):
  """Interface : interface IDENTIFIER Inheritance "{" InterfaceMembers "}" ";"
  """
  p[0] = model.Interface(name=p[2], parent=p[3], members=p[5])