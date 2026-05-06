def _get_json(value):
    """Convert the given value to a JSON object."""
    if hasattr(value, 'replace'):
        value = value.replace('\n', ' ')
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Escape double quotes.
        if hasattr(value, 'replace'):
            value = value.replace('"', '\\"')
        # try putting the value into a string
        return json.loads('"{}"'.format(value))