def render_value(value):
    """Render a value, ensuring that any nested dicts are sorted by key."""
    if isinstance(value, list):
        return '[' + ', '.join(render_value(v) for v in value) + ']'
    elif isinstance(value, dict):
        return (
            '{' +
            ', '.join('{k!r}: {v}'.format(
                k=k, v=render_value(v)) for k, v in sorted(value.items())) +
            '}')
    else:
        return repr(value)