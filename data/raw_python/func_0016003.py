def validate_instance(instance, options=None):
    """Perform STIX JSON Schema validation against STIX input.

    Find the correct schema by looking at the 'type' property of the
    `instance` JSON object.

    Args:
        instance: A Python dictionary representing a STIX object with a
            'type' property.
        options: ValidationOptions instance with validation options for this
            validation run.

    Returns:
        A dictionary of validation results

    """
    if 'type' not in instance:
        raise ValidationError("Input must be an object with a 'type' property.")

    if not options:
        options = ValidationOptions()

    error_gens = []

    # Schema validation
    if instance['type'] == 'bundle' and 'objects' in instance:
        # Validate each object in a bundle separately
        for sdo in instance['objects']:
            if 'type' not in sdo:
                raise ValidationError("Each object in bundle must have a 'type' property.")
            error_gens += _schema_validate(sdo, options)
    else:
        error_gens += _schema_validate(instance, options)

    # Custom validation
    must_checks = _get_musts(options)
    should_checks = _get_shoulds(options)
    output.info("Running the following additional checks: %s."
                % ", ".join(x.__name__ for x in chain(must_checks, should_checks)))
    try:
        errors = _iter_errors_custom(instance, must_checks, options)
        warnings = _iter_errors_custom(instance, should_checks, options)

        if options.strict:
            chained_errors = chain(errors, warnings)
            warnings = []
        else:
            chained_errors = errors
            warnings = [pretty_error(x, options.verbose) for x in warnings]
    except schema_exceptions.RefResolutionError:
        raise SchemaInvalidError('Invalid JSON schema: a JSON reference '
                                 'failed to resolve')

    # List of error generators and message prefixes (to denote which object the
    # error comes from)
    error_gens += [(chained_errors, '')]

    # Prepare the list of errors (this actually triggers the custom validation
    # functions).
    error_list = []
    for gen, prefix in error_gens:
        for error in gen:
            msg = prefix + pretty_error(error, options.verbose)
            error_list.append(SchemaError(msg))

    if error_list:
        valid = False
    else:
        valid = True

    return ObjectValidationResults(is_valid=valid, object_id=instance.get('id', ''),
                                   errors=error_list, warnings=warnings)