def validate(bundle):
    """Validate a bundle object and all of its components.

    The bundle must be passed as a YAML decoded object.

    Return a list of bundle errors, or an empty list if the bundle is valid.
    """
    errors = []
    add_error = errors.append

    # Check that the bundle sections are well formed.
    series, services, machines, relations = _validate_sections(
        bundle, add_error)
    # If there are errors already, there is no point in proceeding with the
    # validation process.
    if errors:
        return errors

    # Validate each individual section.
    _validate_series(series, 'bundle', add_error)
    _validate_services(services, machines, add_error)
    _validate_machines(machines, add_error)
    _validate_relations(relations, services, add_error)

    # Return all the collected errors.
    return errors