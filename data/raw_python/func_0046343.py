def init_farm(farm_name):
    """
    Select a farm to use. This command sets up the replication between your
    local database and the selected cloud server if you have already
    initialized your local database with the `openag db init` command.
    """
    utils.check_for_cloud_server()
    utils.check_for_cloud_user()
    old_farm_name = config["cloud_server"]["farm_name"]
    if old_farm_name and old_farm_name != farm_name:
        raise click.ClickException(
            "Farm \"{}\" already initialized. Run `openag cloud deinit_farm` "
            "to deinitialize it".format(old_farm_name)
        )
    if config["local_server"]["url"]:
        utils.replicate_per_farm_dbs(farm_name=farm_name)
    config["cloud_server"]["farm_name"] = farm_name