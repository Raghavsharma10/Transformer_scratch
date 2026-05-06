def recipe_create(backend, kitchen, name):
    """
    Create a new Recipe
    """
    err_str, use_kitchen = Backend.get_kitchen_from_user(kitchen)
    if use_kitchen is None:
        raise click.ClickException(err_str)
    click.secho("%s - Creating Recipe %s for Kitchen '%s'" % (get_datetime(), name, use_kitchen), fg='green')
    check_and_print(DKCloudCommandRunner.recipe_create(backend.dki, use_kitchen,name))