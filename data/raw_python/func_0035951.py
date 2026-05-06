def kitchen_get(backend, kitchen_name, recipe):
    """
    Get an existing Kitchen
    """
    found_kitchen = DKKitchenDisk.find_kitchen_name()
    if found_kitchen is not None and len(found_kitchen) > 0:
        raise click.ClickException("You cannot get a kitchen into an existing kitchen directory structure.")

    if len(recipe) > 0:
        click.secho("%s - Getting kitchen '%s' and the recipes %s" % (get_datetime(), kitchen_name, str(recipe)), fg='green')
    else:
        click.secho("%s - Getting kitchen '%s'" % (get_datetime(), kitchen_name), fg='green')

    check_and_print(DKCloudCommandRunner.get_kitchen(backend.dki, kitchen_name, os.getcwd(), recipe))