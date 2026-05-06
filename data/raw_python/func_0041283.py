def p_OptionalOrRequiredArgument(p):
  """OptionalOrRequiredArgument : Type Ellipsis IDENTIFIER"""
  p[0] = model.OperationArgument(type=p[1], ellipsis=p[2], name=p[3])