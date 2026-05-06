def set(*args, **kw):
    """Set IRAF environment variables."""

    if len(args) == 0:
        if len(kw) != 0:
            # normal case is only keyword,value pairs
            for keyword, value in kw.items():
                keyword = untranslateName(keyword)
                svalue = str(value)
                _varDict[keyword] = svalue
        else:
            # set with no arguments lists all variables (using same format
            # as IRAF)
            listVars(prefix="    ", equals="=")
    else:
        # The only other case allowed is the peculiar syntax
        # 'set @filename', which only gets used in the zzsetenv.def file,
        # where it reads extern.pkg.  That file also gets read (in full cl
        # mode) by clpackage.cl.  I get errors if I read this during
        # zzsetenv.def, so just ignore it here...
        #
        # Flag any other syntax as an error.
        if (len(args) != 1 or len(kw) != 0 or
                not isinstance(args[0], string_types) or args[0][:1] != '@'):
            raise SyntaxError("set requires name=value pairs")