def subprogram_prototype(vo):
  '''Generate a canonical prototype string
  
  Args:
    vo (VhdlFunction, VhdlProcedure): Subprogram object
  Returns:
    Prototype string.
  '''

  plist = '; '.join(str(p) for p in vo.parameters)

  if isinstance(vo, VhdlFunction):
    if len(vo.parameters) > 0:
      proto = 'function {}({}) return {};'.format(vo.name, plist, vo.return_type)
    else:
      proto = 'function {} return {};'.format(vo.name, vo.return_type)

  else: # procedure
    proto = 'procedure {}({});'.format(vo.name, plist)

  return proto