def subprogram_signature(vo, fullname=None):
  '''Generate a signature string
  
  Args:
    vo (VhdlFunction, VhdlProcedure): Subprogram object
  Returns:
    Signature string.
  '''

  if fullname is None:
    fullname = vo.name

  if isinstance(vo, VhdlFunction):
    plist = ','.join(p.data_type for p in vo.parameters)
    sig = '{}[{} return {}]'.format(fullname, plist, vo.return_type)
  else: # procedure
    plist = ','.join(p.data_type for p in vo.parameters)
    sig = '{}[{}]'.format(fullname, plist)

  return sig