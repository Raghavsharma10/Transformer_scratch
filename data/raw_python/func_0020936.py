def clean_ns(tag):
    """Return a tag and its namespace separately."""
    if '}' in tag:
        split = tag.split('}')
        return split[0].strip('{'), split[-1]
    return '', tag