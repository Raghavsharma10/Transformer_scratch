def vocab_marking_definition(instance):
    """Ensure that the `definition_type` property of `marking-definition`
    objects is one of the values in the STIX 2.0 specification.
    """
    if (instance['type'] == 'marking-definition' and
            'definition_type' in instance and not
            instance['definition_type'] in enums.MARKING_DEFINITION_TYPES):

        return JSONError("Marking definition `definition_type` should be one "
                         "of: %s." % ', '.join(enums.MARKING_DEFINITION_TYPES),
                         instance['id'], 'marking-definition-type')