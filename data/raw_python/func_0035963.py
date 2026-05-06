def order_run(backend, kitchen, recipe, variation, node):
    """
    Run an order: cook a recipe variation
    """
    err_str, use_kitchen = Backend.get_kitchen_from_user(kitchen)
    if use_kitchen is None:
        raise click.ClickException(err_str)
    if recipe is None:
        recipe = DKRecipeDisk.find_recipe_name()
        if recipe is None:
            raise click.ClickException('You must be in a recipe folder, or provide a recipe name.')

    msg = '%s - Create an Order:\n\tKitchen: %s\n\tRecipe: %s\n\tVariation: %s\n' % (get_datetime(), use_kitchen, recipe, variation)
    if node is not None:
        msg += '\tNode: %s\n' % node

    click.secho(msg, fg='green')
    check_and_print(DKCloudCommandRunner.create_order(backend.dki, use_kitchen, recipe, variation, node))