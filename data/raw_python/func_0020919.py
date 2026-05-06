def serializer(metadata_prefix):
    """Return etree_dumper instances.

    :param metadata_prefix: One of the metadata identifiers configured in
        ``OAISERVER_METADATA_FORMATS``.
    """
    metadataFormats = current_app.config['OAISERVER_METADATA_FORMATS']
    serializer_ = metadataFormats[metadata_prefix]['serializer']
    if isinstance(serializer_, tuple):
        return partial(import_string(serializer_[0]), **serializer_[1])
    return import_string(serializer_)