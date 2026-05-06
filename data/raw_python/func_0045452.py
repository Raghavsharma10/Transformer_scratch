def lookup(parser, var, context, resolve=True, apply_filters=True):
    """
    Try to resolve the varialbe in a context
    If ``resolve`` is ``False``, only string variables are returned
    """
    if resolve:
        try:
            return Variable(var).resolve(context)
        except VariableDoesNotExist:
            if apply_filters and var.find('|') > -1:
                return parser.compile_filter(var).resolve(context)
            return Constant(var)
        except TypeError:
            # already resolved
            return var
    return var