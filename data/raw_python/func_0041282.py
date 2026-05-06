def p_OptionalOrRequiredArgument_optional(p):
  """OptionalOrRequiredArgument : optional Type IDENTIFIER Default"""
  p[0] = model.OperationArgument(
    type=p[2], name=p[3], optional=True, default=p[4])