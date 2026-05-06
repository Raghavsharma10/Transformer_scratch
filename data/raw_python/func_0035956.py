def recipe_list(backend, kitchen):
    """
    List the Recipes in a Kitchen
    """
    err_str, use_kitchen = Backend.get_kitchen_from_user(kitchen)
    if use_kitchen is None:
        raise click.ClickException(err_str)
    click.secho("%s - Getting the list of Recipes for Kitchen '%s'" % (get_datetime(), use_kitchen), fg='green')
    check_and_print(DKCloudCommandRunner.list_recipe(backend.dki, use_kitchen))