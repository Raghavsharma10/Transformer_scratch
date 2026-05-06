def _iter_errors_custom(instance, checks, options):
    """Perform additional validation not possible merely with JSON schemas.

    Args:
        instance: The STIX object to be validated.
        checks: A sequence of callables which do the checks.  Each callable
            may be written to accept 1 arg, which is the object to check,
            or 2 args, which are the object and a ValidationOptions instance.
        options: ValidationOptions instance with settings affecting how
            validation should be done.
    """
    # Perform validation
    for v_function in checks:
        try:
            result = v_function(instance)
        except TypeError:
            result = v_function(instance, options)
        if isinstance(result, Iterable):
            for x in result:
                yield x
        elif result is not None:
            yield result

    # Validate any child STIX objects
    for field in instance:
        if type(instance[field]) is list:
            for obj in instance[field]:
                if _is_stix_obj(obj):
                    for err in _iter_errors_custom(obj, checks, options):
                        yield err