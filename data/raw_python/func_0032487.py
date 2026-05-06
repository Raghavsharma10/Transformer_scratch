def registergrant(source=None, setspec=None):
    """Harvest grants from OpenAIRE."""
    with open(source, 'r') as fp:
        data = json.load(fp)
    register_grant(data)