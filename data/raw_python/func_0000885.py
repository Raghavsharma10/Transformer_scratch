def lint_api(api_name, old, new, locations):
    """Lint an acceptable api metadata."""
    is_new_api = not old
    api_location = locations['api']
    changelog = new.get('changelog', {})
    changelog_location = api_location

    if locations['changelog']:
        changelog_location = list(locations['changelog'].values())[0]

    # apis must have documentation if they are new
    if not new.get('doc'):
        msg_type = LintError if is_new_api else LintWarning
        yield msg_type(
            'doc',
            'missing docstring documentation',
            api_name=api_name,
            location=locations.get('view', api_location)
        )

    introduced_at = new.get('introduced_at')
    if introduced_at is None:
        yield LintError(
            'introduced_at',
            'missing introduced_at field',
            location=api_location,
        )

    if not is_new_api:
        # cannot change introduced_at if we already have it
        old_introduced_at = old.get('introduced_at')
        if old_introduced_at is not None:
            if old_introduced_at != introduced_at:
                yield LintError(
                    'introduced_at',
                    'introduced_at changed from {} to {}',
                    old_introduced_at,
                    introduced_at,
                    api_name=api_name,
                    location=api_location,
                )

    # cannot change url
    if new['url'] != old.get('url', new['url']):
        yield LintError(
            'url',
            'url changed from {} to {}',
            old['url'],
            new['url'],
            api_name=api_name,
            location=api_location,
        )

    # cannot add required fields
    for removed in set(old.get('methods', [])) - set(new['methods']):
        yield LintError(
            'methods',
            'HTTP method {} removed',
            removed,
            api_name=api_name,
            location=api_location,
        )

    for schema in ['request_schema', 'response_schema']:
        new_schema = new.get(schema)
        if new_schema is None:
            continue

        schema_location = locations[schema]
        old_schema = old.get(schema, {})

        for message in walk_schema(
                schema, old_schema, new_schema, root=True, new_api=is_new_api):
            if isinstance(message, CheckChangelog):
                if message.revision not in changelog:
                    yield LintFixit(
                        message.name,
                        'No changelog entry for revision {}',
                        message.revision,
                        location=changelog_location,
                    )
            else:
                # add in here, saves passing it down the recursive call
                message.location = schema_location
                yield message