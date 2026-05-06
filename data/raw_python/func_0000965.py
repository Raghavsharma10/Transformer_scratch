def extract_schemas_from_file(source_path):
    """Extract schemas from 'source_path'.

    :returns: a list of ViewSchema objects on success, None if no schemas
        could be extracted.
    """
    logging.info("Extracting schemas from %s", source_path)
    try:
        with open(source_path, 'r') as source_file:
            source = source_file.read()
    except (FileNotFoundError, PermissionError) as e:
        logging.error("Cannot extract schemas: %s", e.strerror)
    else:
        try:
            schemas = extract_schemas_from_source(source, source_path)
        except SyntaxError as e:
            logging.error("Cannot extract schemas: %s", str(e))
        else:
            logging.info(
                "Extracted %d %s",
                len(schemas),
                "schema" if len(schemas) == 1 else "schemas")
            return schemas