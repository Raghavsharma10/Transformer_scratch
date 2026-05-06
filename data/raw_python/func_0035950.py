def kitchen_list(backend):
    """
    List all Kitchens
    """
    click.echo(click.style('%s - Getting the list of kitchens' % get_datetime(), fg='green'))
    check_and_print(DKCloudCommandRunner.list_kitchen(backend.dki))