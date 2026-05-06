def parse_unit(expression):
    """Evaluate a python expression string containing constants

       Argument:
        | ``expression``  --  A string containing a numerical expressions
                              including unit conversions.

       In addition to the variables in this module, also the following
       shorthands are supported:

    """
    try:
        g = globals()
        g.update(shorthands)
        return float(eval(str(expression), g))
    except:
        raise ValueError("Could not interpret '%s' as a unit or a measure." % expression)