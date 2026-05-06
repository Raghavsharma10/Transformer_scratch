def file_add(backend, kitchen, recipe, message, filepath):
    """
    Add a newly created file to a Recipe
    """
    err_str, use_kitchen = Backend.get_kitchen_from_user(kitchen)
    if use_kitchen is None:
        raise click.ClickException(err_str)
    if recipe is None:
        recipe = DKRecipeDisk.find_recipe_name()
        if recipe is None:
            raise click.ClickException('You must be in a recipe folder, or provide a recipe name.')

    click.secho('%s - Adding File (%s) to Recipe (%s) in kitchen(%s) with message (%s)' %
                (get_datetime(), filepath, recipe, use_kitchen, message), fg='green')
    check_and_print(DKCloudCommandRunner.add_file(backend.dki, use_kitchen, recipe, message, filepath))