def _validate_mapping(index, strict=False):
    """Check that an index mapping JSON file exists."""
    try:
        settings.get_index_mapping(index)
    except IOError:
        if strict:
            raise ImproperlyConfigured("Index '%s' has no mapping file." % index)
        else:
            logger.warning("Index '%s' has no mapping, relying on ES instead.", index)