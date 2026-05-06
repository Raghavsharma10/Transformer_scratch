def copy_resume(src_uri, dest_base_uri, config_path=None, progressbar=None):
    """Resume coping a dataset to another location.

    Items that have been copied to the destination and have the same size
    as in the source dataset are skipped. All other items are copied across
    and the dataset is frozen.

    :param src_uri: URI of dataset to be copied
    :param dest_base_uri: base of URI for copy target
    :param config_path: path to dtool configuration file
    :returns: URI of new dataset
    """
    dataset = DataSet.from_uri(src_uri)

    # Generate the URI of the destination proto dataset.
    dest_uri = _generate_uri(dataset._admin_metadata, dest_base_uri)

    proto_dataset = ProtoDataSet.from_uri(dest_uri)

    _copy_content(dataset, proto_dataset, progressbar)
    proto_dataset.freeze(progressbar=progressbar)

    return proto_dataset.uri