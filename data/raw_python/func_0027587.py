def assert_instance_of(value,
                       allowed_types  # type: Union[Type, Tuple[Type]]
                       ):
    """
    An inlined version of instance_of(var_types)(value) without 'return True': it does not return anything in case of
    success, and raises a HasWrongType exception in case of failure.

    Used in validate and validation/validator

    :param value: the value to check
    :param allowed_types: the type(s) to enforce. If a tuple of types is provided it is considered alternate types: one
        match is enough to succeed. If None, type will not be enforced
    :return:
    """
    if not isinstance(value, allowed_types):
        try:
            # more than 1 ?
            allowed_types[1]
            raise HasWrongType(wrong_value=value, ref_type=allowed_types,
                               help_msg='Value should be an instance of any of {ref_type}')
        except IndexError:
            # 1
            allowed_types = allowed_types[0]
        except TypeError:
            # 1
            pass
        raise HasWrongType(wrong_value=value, ref_type=allowed_types)