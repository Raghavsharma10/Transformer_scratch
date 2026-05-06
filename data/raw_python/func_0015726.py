def build_docstring(func_name, args, ret, throws, signal_owner_type=None):
    """Create a docstring in the form:
        name(in_name: type) -> (ret_type, out_name: type)
    """

    out_args = []
    if ret and not ret.ignore:
        if ret.py_type is None:
            out_args.append("unknown")
        else:
            tname = get_type_name(ret.py_type)
            if ret.may_return_null:
                tname += " or None"
            out_args.append(tname)

    in_args = []

    if signal_owner_type is not None:
        name = get_signal_owner_var_name(signal_owner_type)
        in_args.append("%s: %s" % (
            name, get_type_name(signal_owner_type.pytype)))

    for arg in args:
        if arg.is_aux:
            continue

        if arg.is_direction_in():
            if arg.py_type is None:
                in_args.append(arg.in_var)
            else:
                tname = get_type_name(arg.py_type)
                if arg.may_be_null:
                    tname += " or None"
                in_args.append("%s: %s" % (arg.in_var, tname))

        if arg.is_direction_out():
            if arg.py_type is None:
                out_args.append(arg.name)
            else:
                tname = get_type_name(arg.py_type)
                # if may_be_null means the arg is nullable, it is nullable
                # and the marshalling returns None for a NULL pointer
                if may_be_null_is_nullable() and arg.may_be_null and \
                        arg.can_unpack_none:
                    tname += " or None"
                # When can we assume that out args return None?
                out_args.append("%s: %s" % (arg.name, tname))

    in_def = ", ".join(in_args)

    if not out_args:
        out_def = "None"
    elif len(out_args) == 1:
        out_def = out_args[0]
    else:
        out_def = "(%s)" % ", ".join(out_args)

    error = ""
    if throws:
        error = "raises "

    return "%s(%s) %s-> %s" % (func_name, in_def, error, out_def)