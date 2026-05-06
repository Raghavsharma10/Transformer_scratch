def _get_shoulds(options):
    """Return the list of 'SHOULD' validators for the correct version of STIX.

    Args:
        options: ValidationOptions instance with validation options for this
            validation run, including the STIX spec version.
    """
    if options.version == '2.0':
        return shoulds20.list_shoulds(options)
    else:
        return shoulds21.list_shoulds(options)