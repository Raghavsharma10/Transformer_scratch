def add_base_type_dynamically(error_type, additional_type):
    """
    Utility method to create a new type dynamically, inheriting from both error_type (first) and additional_type
    (second). The class representation (repr(cls)) of the resulting class reflects this by displaying both names
    (fully qualified for the first type, __name__ for the second)

    For example
    ```
    > new_type = add_base_type_dynamically(ValidationError, ValueError)
    > repr(new_type)
    "<class 'valid8.entry_points.ValidationError+ValueError'>"
    ```
    :return:
    """
    # the new type created dynamically, with the same name
    class new_error_type(with_metaclass(MetaReprForValidator, error_type, additional_type, object)):
        pass

    new_error_type.__name__ = error_type.__name__ + '[' + additional_type.__name__ + ']'
    if sys.version_info >= (3, 0):
        new_error_type.__qualname__ = error_type.__qualname__ + '[' + additional_type.__qualname__+ ']'
    new_error_type.__module__ = error_type.__module__

    return new_error_type