def kitchen_delete(backend, kitchen):
    """
    Provide the name of the kitchen to delete
    """
    click.secho('%s - Deleting kitchen %s' % (get_datetime(), kitchen), fg='green')
    master = 'master'
    if kitchen.lower() != master.lower():
        check_and_print(DKCloudCommandRunner.delete_kitchen(backend.dki, kitchen))
    else:
        raise click.ClickException('Cannot delete the kitchen called %s' % master)