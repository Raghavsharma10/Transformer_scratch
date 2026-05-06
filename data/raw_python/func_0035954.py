def kitchen_config(backend, kitchen, add, get, unset, listall):
    """
    Get and Set Kitchen variable overrides
    """
    err_str, use_kitchen = Backend.get_kitchen_from_user(kitchen)
    if use_kitchen is None:
        raise click.ClickException(err_str)
    check_and_print(DKCloudCommandRunner.config_kitchen(backend.dki, use_kitchen, add, get, unset, listall))