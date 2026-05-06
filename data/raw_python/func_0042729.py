def get_schema_from_list(table_name, frum):
    """
    SCAN THE LIST FOR COLUMN TYPES
    """
    columns = UniqueIndex(keys=("name",))
    _get_schema_from_list(frum, ".", parent=".", nested_path=ROOT_PATH, columns=columns)
    return Schema(table_name=table_name, columns=list(columns))