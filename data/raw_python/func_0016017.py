def relationships_strict(instance):
    """Ensure that only the relationship types defined in the specification are
    used.
    """
    # Don't check objects that aren't relationships or that are custom objects
    if (instance['type'] != 'relationship' or
            instance['type'] not in enums.TYPES):
        return

    if ('relationship_type' not in instance or 'source_ref' not in instance or
            'target_ref' not in instance):
        # Since these fields are required, schemas will already catch the error
        return

    r_type = instance['relationship_type']
    try:
        r_source = re.search(r"(.+)\-\-", instance['source_ref']).group(1)
        r_target = re.search(r"(.+)\-\-", instance['target_ref']).group(1)
    except (AttributeError, TypeError):
        # Schemas already catch errors of these properties not being strings or
        # not containing the string '--'.
        return

    if (r_type in enums.COMMON_RELATIONSHIPS or
            r_source in enums.NON_SDOS or
            r_target in enums.NON_SDOS):
        # If all objects can have this relationship type, no more checks needed
        # Schemas already catch if source/target type cannot have relationship
        return

    if r_source not in enums.RELATIONSHIPS:
        return JSONError("'%s' is not a suggested relationship source object "
                         "for the '%s' relationship." % (r_source, r_type),
                         instance['id'], 'relationship-types')

    if r_type not in enums.RELATIONSHIPS[r_source]:
        return JSONError("'%s' is not a suggested relationship type for '%s' "
                         "objects." % (r_type, r_source), instance['id'],
                         'relationship-types')

    if r_target not in enums.RELATIONSHIPS[r_source][r_type]:
        return JSONError("'%s' is not a suggested relationship target object "
                         "for '%s' objects with the '%s' relationship."
                         % (r_target, r_source, r_type), instance['id'],
                         'relationship-types')