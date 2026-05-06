def secret_write(backend,entry):
    """
    Write a secret
    """
    path,value=entry.split('=')

    if value.startswith('@'):
        with open(value[1:]) as vfile:
            value = vfile.read()

    click.echo(click.style('%s - Writing secret' % get_datetime(), fg='green'))
    check_and_print(
        DKCloudCommandRunner.secret_write(backend.dki,path,value))