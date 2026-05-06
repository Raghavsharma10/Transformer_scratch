def pretty_json(data):
    """Return a pretty formatted json
    """
    data = json.loads(data.decode('utf-8'))
    return json.dumps(data, indent=4, sort_keys=True)