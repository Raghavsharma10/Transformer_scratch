def recipe_compile(backend, kitchen, recipe, variation):
    """
    Apply variables to a Recipe
    """
    err_str, use_kitchen = Backend.get_kitchen_from_user(kitchen)
    if use_kitchen is None:
        raise click.ClickException(err_str)

    if recipe is None:
        recipe = DKRecipeDisk.find_recipe_name()
        if recipe is None:
            raise click.ClickException('You must be in a recipe folder, or provide a recipe name.')

    click.secho('%s - Get the Compiled OrderRun of Recipe %s.%s in Kitchen %s' % (get_datetime(), recipe, variation, use_kitchen),
                fg='green')
    check_and_print(DKCloudCommandRunner.get_compiled_serving(backend.dki, use_kitchen, recipe, variation))