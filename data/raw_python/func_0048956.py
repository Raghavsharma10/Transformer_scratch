def _validate_sections(bundle, add_error):
    """Check that the base bundle sections are valid.

    The bundle argument is a YAML decoded bundle content.

    A bundle is composed of series, services, machines and relations.
    Only the services section is mandatory.

    Use the given add_error callable to register validation error.
    Return the four sections
    """
    # Check that the bundle itself is well formed.
    if not isdict(bundle):
        add_error('bundle does not appear to be a bundle')
        return None, None, None, None
    # Validate the services section.
    services = bundle.get('services', {})
    if not services:
        add_error('bundle does not define any services')
    elif not isdict(services):
        add_error('services spec does not appear to be well-formed')
    # Validate the machines section.
    machines = bundle.get('machines')
    if machines is not None:
        if isdict(machines):
            try:
                machines = dict((int(k), v) for k, v in machines.items())
            except (TypeError, ValueError):
                add_error('machines spec identifiers must be digits')
        else:
            add_error('machines spec does not appear to be well-formed')
    # Validate the relations section.
    relations = bundle.get('relations')
    if (relations is not None) and (not islist(relations)):
        add_error('relations spec does not appear to be well-formed')
    return bundle.get('series'), services, machines, relations