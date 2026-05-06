def generate_proto_dataset(admin_metadata, base_uri, config_path=None):
    """Return :class:`dtoolcore.ProtoDataSet` instance.

    :param admin_metadata: dataset administrative metadata
    :param base_uri: base URI for proto dataset
    :param config_path: path to dtool configuration file
    """
    uri = _generate_uri(admin_metadata, base_uri)
    return ProtoDataSet(uri, admin_metadata, config_path)