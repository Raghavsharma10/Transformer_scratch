def recipe_get(backend, recipe):
    """
    Get the latest files for this recipe.
    """
    recipe_root_dir = DKRecipeDisk.find_recipe_root_dir()
    if recipe_root_dir is None:
        if recipe is None:
            raise click.ClickException("\nPlease change to a recipe folder or provide a recipe name arguement")

        # raise click.ClickException('You must be in a Recipe folder')
        kitchen_root_dir = DKKitchenDisk.is_kitchen_root_dir()
        if not kitchen_root_dir:
            raise click.ClickException("\nPlease change to a recipe folder or a kitchen root dir.")
        recipe_name = recipe
        start_dir = DKKitchenDisk.find_kitchen_root_dir()
    else:
        recipe_name = DKRecipeDisk.find_recipe_name()
        if recipe is not None:
            if recipe_name != recipe:
                raise click.ClickException("\nThe recipe name argument '%s' is inconsistent with the current directory '%s'" % (recipe, recipe_root_dir))
        start_dir = recipe_root_dir

    kitchen_name = Backend.get_kitchen_name_soft()
    click.secho("%s - Getting the latest version of Recipe '%s' in Kitchen '%s'" % (get_datetime(), recipe_name, kitchen_name), fg='green')
    check_and_print(DKCloudCommandRunner.get_recipe(backend.dki, kitchen_name, recipe_name, start_dir))