def load_json_from_string(string):
    """Load schema from JSON string"""
    try:
        json_data = json.loads(string)
    except ValueError as e:
        raise ValueError('Given string is not valid JSON: {}'.format(e))
    else:
        return json_data