def copy(src_uri, dest_base_uri, config_path=None, progressbar=None):
    """Copy a dataset to another location.

    :param src_uri: URI of dataset to be copied
    :param dest_base_uri: base of URI for copy target
    :param config_path: path to dtool configuration file
    :returns: URI of new dataset
    """
    dataset = DataSet.from_uri(src_uri)

    proto_dataset = _copy_create_proto_dataset(
        dataset,
        dest_base_uri,
        config_path,
        progressbar
    )
    _copy_content(dataset, proto_dataset, progressbar)
    proto_dataset.freeze(progressbar=progressbar)

    return proto_dataset.uri