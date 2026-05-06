def validate_field(cls,
                   field_name,
                   *validation_func,  # type: ValidationFuncs
                   **kwargs):
    # type: (...) -> Callable
    """
    A class decorator. It goes through all class variables and for all of those that are descriptors with a __set__,
    it wraps the descriptors' setter function with a `validate_arg` annotation

    :param field_name:
    :param validation_func:
    :param help_msg:
    :param error_type:
    :param none_policy:
    :param kw_context_args:
    :return
    """
    return decorate_cls_with_validation(cls, field_name, *validation_func, **kwargs)