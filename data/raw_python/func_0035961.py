def file_update_all(backend, message, dryrun):
    """
    Update all of the changed files for this Recipe
    """
    kitchen = DKCloudCommandRunner.which_kitchen_name()
    if kitchen is None:
        raise click.ClickException('You must be in a Kitchen')
    recipe_dir = DKRecipeDisk.find_recipe_root_dir()
    if recipe_dir is None:
        raise click.ClickException('You must be in a Recipe folder')
    recipe = DKRecipeDisk.find_recipe_name()

    if dryrun:
        click.secho('%s - Display all changed files in Recipe (%s) in Kitchen(%s) with message (%s)' %
                    (get_datetime(), recipe, kitchen, message), fg='green')
    else:
        click.secho('%s - Updating all changed files in Recipe (%s) in Kitchen(%s) with message (%s)' %
                    (get_datetime(), recipe, kitchen, message), fg='green')
    check_and_print(DKCloudCommandRunner.update_all_files(backend.dki, kitchen, recipe, recipe_dir, message, dryrun))