def show_account():
    """
    Exports current account configuration in
    shell-friendly form. Takes into account
    explicit top-level flags like --organization.
    """
    click.echo("# tonomi api")
    for (key, env) in REVERSE_MAPPING.items():
        value = QUBELL.get(key, None)
        if value:
            click.echo("export %s='%s'" % (env, value))
    if any(map(lambda x: PROVIDER.get(x), REVERSE_PROVIDER_MAPPING.keys())):
        click.echo("# cloud account")
        for (key, env) in REVERSE_PROVIDER_MAPPING.items():
            value = PROVIDER.get(key, None)
            if value:
                click.echo("export %s='%s'" % (env, value))