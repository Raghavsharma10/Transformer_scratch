def should_be_hidden_as_cause(exc):
    """ Used everywhere to decide if some exception type should be displayed or hidden as the casue of an error """
    # reduced traceback in case of HasWrongType (instance_of checks)
    from valid8.validation_lib.types import HasWrongType, IsWrongType
    return isinstance(exc, (HasWrongType, IsWrongType))