def fetch_and_transform(
        transformed_filename,
        transformer,
        loader,
        source_filename,
        source_url,
        subdir=None):
    """
    Fetch a remote file from `source_url`, save it locally as `source_filename` and then use
    the `loader` and `transformer` function arguments to turn this saved data into an in-memory
    object.
    """
    transformed_path = build_path(transformed_filename, subdir)
    if not os.path.exists(transformed_path):
        source_path = fetch_file(source_url, source_filename, subdir)
        logger.info("Generating data file %s from %s", transformed_path, source_path)
        result = transformer(source_path, transformed_path)
    else:
        logger.info("Cached data file: %s", transformed_path)
        result = loader(transformed_path)
    assert os.path.exists(transformed_path)
    return result