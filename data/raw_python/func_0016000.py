def _get_musts(options):
    """Return the list of 'MUST' validators for the correct version of STIX.

    Args:
        options: ValidationOptions instance with validation options for this
            validation run, including the STIX spec version.
    """
    if options.version == '2.0':
        return musts20.list_musts(options)
    else:
        return musts21.list_musts(options)