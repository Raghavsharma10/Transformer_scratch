def list_musts(options):
    """Construct the list of 'MUST' validators to be run by the validator.
    """
    validator_list = [
        timestamp,
        timestamp_compare,
        observable_timestamp_compare,
        object_marking_circular_refs,
        granular_markings_circular_refs,
        marking_selector_syntax,
        observable_object_references,
        artifact_mime_type,
        character_set,
        language,
        software_language,
        patterns,
        language_contents,
    ]

    # --strict-types
    if options.strict_types:
        validator_list.append(types_strict)

    # --strict-properties
    if options.strict_properties:
        validator_list.append(properties_strict)

    return validator_list