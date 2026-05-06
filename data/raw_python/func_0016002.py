def _schema_validate(sdo, options):
    """Set up validation of a single STIX object against its type's schema.
    This does no actual validation; it just returns generators which must be
    iterated to trigger the actual generation.

    This function first creates generators for the built-in schemas, then adds
    generators for additional schemas from the options, if specified.

    Do not call this function directly; use validate_instance() instead, as it
    calls this one. This function does not perform any custom checks.
    """
    error_gens = []

    if 'id' in sdo:
        try:
            error_prefix = sdo['id'] + ": "
        except TypeError:
            error_prefix = 'unidentifiable object: '
    else:
        error_prefix = ''

    # Get validator for built-in schema
    base_sdo_errors = _get_error_generator(sdo['type'], sdo, version=options.version)
    if base_sdo_errors:
        error_gens.append((base_sdo_errors, error_prefix))

    # Get validator for any user-supplied schema
    if options.schema_dir:
        custom_sdo_errors = _get_error_generator(sdo['type'], sdo, options.schema_dir)
        if custom_sdo_errors:
            error_gens.append((custom_sdo_errors, error_prefix))

    # Validate each cyber observable object separately
    if sdo['type'] == 'observed-data' and 'objects' in sdo:
        # Check if observed data property is in dictionary format
        if not isinstance(sdo['objects'], dict):
            error_gens.append(([schema_exceptions.ValidationError("Observed Data objects must be in dict format.", error_prefix)],
                              error_prefix))
            return error_gens

        for key, obj in iteritems(sdo['objects']):
            if 'type' not in obj:
                error_gens.append(([schema_exceptions.ValidationError("Observable object must contain a 'type' property.", error_prefix)],
                                   error_prefix + 'object \'' + key + '\': '))
                continue
            # Get validator for built-in schemas
            base_obs_errors = _get_error_generator(obj['type'],
                                                   obj,
                                                   None,
                                                   options.version,
                                                   'cyber-observable-core')
            if base_obs_errors:
                error_gens.append((base_obs_errors,
                                   error_prefix + 'object \'' + key + '\': '))

            # Get validator for any user-supplied schema
            custom_obs_errors = _get_error_generator(obj['type'],
                                                     obj,
                                                     options.schema_dir,
                                                     options.version,
                                                     'cyber-observable-core')
            if custom_obs_errors:
                error_gens.append((custom_obs_errors,
                                   error_prefix + 'object \'' + key + '\': '))

    return error_gens