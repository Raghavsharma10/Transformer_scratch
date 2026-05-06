def metadata_lint(old, new, locations):
    """Run the linter over the new metadata, comparing to the old."""
    # ensure we don't modify the metadata
    old = old.copy()
    new = new.copy()
    # remove version info
    old.pop('$version', None)
    new.pop('$version', None)

    for old_group_name in old:
        if old_group_name not in new:
            yield LintError('', 'api group removed', api_name=old_group_name)

    for group_name, new_group in new.items():
        old_group = old.get(group_name, {'apis': {}})

        for name, api in new_group['apis'].items():
            old_api = old_group['apis'].get(name, {})
            api_locations = locations[name]
            for message in lint_api(name, old_api, api, api_locations):
                message.api_name = name
                if message.location is None:
                    message.location = api_locations['api']
                yield message