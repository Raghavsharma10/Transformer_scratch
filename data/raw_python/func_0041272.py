def p_Exception(p):
  """Exception : exception IDENTIFIER Inheritance "{" ExceptionMembers "}" ";"
  """
  p[0] = model.Exception(name=p[2], parent=p[3], members=p[5])