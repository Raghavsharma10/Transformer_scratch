def secret_list(backend,path):
    """
    List all Secrets
    """
    click.echo(click.style('%s - Getting the list of secrets' % get_datetime(), fg='green'))
    check_and_print(
        DKCloudCommandRunner.secret_list(backend.dki,path))