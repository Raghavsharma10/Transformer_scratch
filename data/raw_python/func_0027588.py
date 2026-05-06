def assert_subclass_of(typ,
                       allowed_types  # type: Union[Type, Tuple[Type]]
                       ):
    """
    An inlined version of subclass_of(var_types)(value) without 'return True': it does not return anything in case of
    success, and raises a IsWrongType exception in case of failure.

    Used in validate and validation/validator

    :param typ: the type to check
    :param allowed_types: the type(s) to enforce. If a tuple of types is provided it is considered alternate types:
        one match is enough to succeed. If None, type will not be enforced
    :return:
    """
    if not issubclass(typ, allowed_types):
        try:
            # more than 1 ?
            allowed_types[1]
            raise IsWrongType(wrong_value=typ, ref_type=allowed_types,
                              help_msg='Value should be a subclass of any of {ref_type}')
        except IndexError:
            # 1
            allowed_types = allowed_types[0]
        except TypeError:
            # 1
            pass
        raise IsWrongType(wrong_value=typ, ref_type=allowed_types)