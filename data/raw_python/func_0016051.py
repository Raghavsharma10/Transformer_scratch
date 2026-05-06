def patterns(instance, options):
    """Ensure that the syntax of the pattern of an indicator is valid, and that
    objects and properties referenced by the pattern are valid.
    """
    if instance['type'] != 'indicator' or 'pattern' not in instance:
        return

    pattern = instance['pattern']
    if not isinstance(pattern, string_types):
        return  # This error already caught by schemas
    errors = pattern_validator(pattern)

    # Check pattern syntax
    if errors:
        for e in errors:
            yield PatternError(str(e), instance['id'])
        return

    type_format_re = re.compile(r'^\-?[a-z0-9]+(-[a-z0-9]+)*\-?$')
    property_format_re = re.compile(r'^[a-z0-9_]{3,250}$')

    p = Pattern(pattern)
    inspection = p.inspect().comparisons
    for objtype in inspection:
        # Check observable object types
        if objtype in enums.OBSERVABLE_TYPES:
            pass
        elif options.strict_types:
            yield PatternError("'%s' is not a valid STIX observable type"
                               % objtype, instance['id'])
        elif (not type_format_re.match(objtype) or
              len(objtype) < 3 or len(objtype) > 250):
            yield PatternError("'%s' is not a valid observable type name"
                               % objtype, instance['id'])
        elif (all(x not in options.disabled for x in ['all', 'format-checks', 'custom-prefix']) and
              not CUSTOM_TYPE_PREFIX_RE.match(objtype)):
            yield PatternError("Custom Observable Object type '%s' should start "
                               "with 'x-' followed by a source unique identifier "
                               "(like a domain name with dots replaced by "
                               "hyphens), a hyphen and then the name"
                               % objtype, instance['id'])
        elif (all(x not in options.disabled for x in ['all', 'format-checks', 'custom-prefix-lax']) and
              not CUSTOM_TYPE_LAX_PREFIX_RE.match(objtype)):
            yield PatternError("Custom Observable Object type '%s' should start "
                               "with 'x-'" % objtype, instance['id'])

        # Check observable object properties
        expression_list = inspection[objtype]
        for exp in expression_list:
            path = exp[0]
            # Get the property name without list index, dictionary key, or referenced object property
            prop = path[0]
            if objtype in enums.OBSERVABLE_PROPERTIES and prop in enums.OBSERVABLE_PROPERTIES[objtype]:
                continue
            elif options.strict_properties:
                yield PatternError("'%s' is not a valid property for '%s' objects"
                                   % (prop, objtype), instance['id'])
            elif not property_format_re.match(prop):
                yield PatternError("'%s' is not a valid observable property name"
                                   % prop, instance['id'])
            elif (all(x not in options.disabled for x in ['all', 'format-checks', 'custom-prefix']) and
                  not CUSTOM_PROPERTY_PREFIX_RE.match(prop)):
                yield PatternError("Cyber Observable Object custom property '%s' "
                                   "should start with 'x_' followed by a source "
                                   "unique identifier (like a domain name with "
                                   "dots replaced by underscores), an "
                                   "underscore and then the name"
                                   % prop, instance['id'])
            elif (all(x not in options.disabled for x in ['all', 'format-checks', 'custom-prefix-lax']) and
                  not CUSTOM_PROPERTY_LAX_PREFIX_RE.match(prop)):
                yield PatternError("Cyber Observable Object custom property '%s' "
                                   "should start with 'x_'" % prop, instance['id'])