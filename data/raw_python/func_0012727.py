def normalize_topic(topic):
    """
    Get a canonical representation of a Wikipedia topic, which may include
    a disambiguation string in parentheses.

    Returns (name, disambig), where "name" is the normalized topic name,
    and "disambig" is a string corresponding to the disambiguation text or
    None.
    """
    # find titles of the form Foo (bar)
    topic = topic.replace('_', ' ')
    match = re.match(r'([^(]+) \(([^)]+)\)', topic)
    if not match:
        return normalize(topic), None
    else:
        return normalize(match.group(1)), 'n/' + match.group(2).strip(' _')