def p_Typedef(p):
  """Typedef : typedef ExtendedAttributeList Type IDENTIFIER ";"
  """
  p[0] = model.Typedef(type_extended_attributes=p[2], type=p[3], name=p[4])