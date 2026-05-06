def normalize_cell_value(value):
    """Process value for writing into a cell.

    Args:
        value: any type of variable

    Returns:
        json serialized value if value is list or dict, else value
    """
    if isinstance(value, dict) or isinstance(value, list):
        return json.dumps(value)
    return value