def _expand1(instring, noerror):
    """Expand a string with embedded IRAF variables (IRAF virtual filename)."""

    # first expand names in parentheses
    # note this works on nested names too, expanding from the
    # inside out (just like IRAF)
    mm = __re_var_paren.search(instring)
    while mm is not None:
        # remove embedded dollar signs from name
        varname = mm.group('varname').replace('$','')
        if defvar(varname):
            varname = envget(varname)
        elif noerror:
            varname = ""
        else:
            raise ValueError("Undefined variable `%s' in string `%s'" %
                             (varname, instring))

        instring = instring[:mm.start()] + varname + instring[mm.end():]
        mm = __re_var_paren.search(instring)
    # now expand variable name at start of string
    mm = __re_var_match.match(instring)
    if mm is None:
        return instring
    varname = mm.group('varname')
    if varname in ['', ' ', None]:
        mm = __re_var_match2.match(instring)
        varname = mm.group('varname')

    if defvar(varname):
        # recursively expand string after substitution
        return _expand1(envget(varname) + instring[mm.end():], noerror)
    elif noerror:
        return _expand1(varname + instring[mm.end():], noerror)
    else:
        raise ValueError("Undefined variable `%s' in string `%s'" %
                         (varname, instring))