def find_schema(schema_dir, obj_type):
    """Search the `schema_dir` directory for a schema called `obj_type`.json.
    Return the file path of the first match it finds.
    """
    schema_filename = obj_type + '.json'

    for root, dirnames, filenames in os.walk(schema_dir):
        if schema_filename in filenames:
            return os.path.join(root, schema_filename)