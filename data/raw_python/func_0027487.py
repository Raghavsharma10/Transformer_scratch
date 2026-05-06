def subclass_of(*args):
    """
    This type validation function can be used in two modes:
     * providing two arguments (c, ref_type), it returns `True` if issubclass(c, ref_type) and raises a IsWrongType
     error if not. If ref_type is a set of types, any match with one of the included types will do
     * providing a single argument (ref_type), this is a function generator. It returns a validation function to check
     that `subclass_of(c, ref_type)`.

    :param args:
    :return:
    """
    if len(args) == 2:
        # Standard mode
        typ, ref_type = args
        if not isinstance(ref_type, set):
            # ref_type is a single type
            if issubclass(typ, ref_type):
                return True
            else:
                raise IsWrongType(wrong_value=typ, ref_type=ref_type)
        else:
            # ref_type is a set
            match = False
            # test against each of the provided types
            for ref in ref_type:
                if issubclass(typ, ref):
                    match = True
                    break
            if match:
                return True
            else:
                raise IsWrongType(wrong_value=typ, ref_type=ref_type,
                                  help_msg='Value should be a subclass of any of {ref_type}')

    elif len(args) == 1:
        # Function generator mode
        ref_type = args[0]
        if not isinstance(ref_type, set):
            def subclass_of_ref(x):
                if issubclass(x, ref_type):
                    return True
                else:
                    raise IsWrongType(wrong_value=x, ref_type=ref_type)
        else:
            # ref_type is a set
            def subclass_of_ref(x):
                match = False
                # test against each of the provided types
                for ref in ref_type:
                    if issubclass(x, ref):
                        match = True
                        break
                if match:
                    return True
                else:
                    raise IsWrongType(wrong_value=x, ref_type=ref_type,
                                      help_msg='Value should be a subclass of any of {ref_type}')

        subclass_of_ref.__name__ = 'subclass_of_{}'.format(ref_type)
        return subclass_of_ref
    else:
        raise TypeError('subclass_of expected 2 (normal) or 1 (function generator) arguments, got ' + str(len(args)))