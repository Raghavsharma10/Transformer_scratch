def file_resolve(backend, filepath):
    """
    Mark a conflicted file as resolved, so that a merge can be completed
    """
    recipe = DKRecipeDisk.find_recipe_name()
    if recipe is None:
        raise click.ClickException('You must be in a recipe folder.')

    click.secho("%s - Resolving conflicts" % get_datetime())

    for file_to_resolve in filepath:
        if not os.path.exists(file_to_resolve):
            raise click.ClickException('%s does not exist' % file_to_resolve)
        check_and_print(DKCloudCommandRunner.resolve_conflict(file_to_resolve))