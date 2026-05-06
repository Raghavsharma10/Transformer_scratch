def urlmap(patterns):
    """Recursively build a map of (group, name) => url patterns.

    Group is either the resolver namespace or app name for the url config.

    The urls are joined with any prefixes, and cleaned up of extraneous regex
    specific syntax."""
    for pattern in patterns:
        group = getattr(pattern, 'namespace', None)
        if group is None:
            group = getattr(pattern, 'app_name', None)
        path = '/' + get_pattern(pattern).lstrip('^').rstrip('$')
        if isinstance(pattern, PATTERNS):
            yield (group, pattern.name), path
        elif isinstance(pattern, RESOLVERS):
            subpatterns = pattern.url_patterns
            for (_, name), subpath in urlmap(subpatterns):
                yield (group, name), path.rstrip('/') + subpath