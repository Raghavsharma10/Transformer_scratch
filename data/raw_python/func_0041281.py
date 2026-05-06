def p_OperationRest(p):
  """OperationRest : ReturnType OptionalIdentifier "(" ArgumentList ")" ";"
  """
  p[0] = model.Operation(return_type=p[1], name=p[2], arguments=p[4])