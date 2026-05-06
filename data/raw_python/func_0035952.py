def kitchen_create(backend, parent, kitchen):
    """
    Create a new kitchen
    """
    click.secho('%s - Creating kitchen %s from parent kitchen %s' % (get_datetime(), kitchen, parent), fg='green')
    master = 'master'
    if kitchen.lower() != master.lower():
        check_and_print(DKCloudCommandRunner.create_kitchen(backend.dki, parent, kitchen))
    else:
        raise click.ClickException('Cannot create a kitchen called %s' % master)