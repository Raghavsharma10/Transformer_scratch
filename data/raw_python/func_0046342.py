def list_farms():
    """
    List all farms you can manage. If you have selected a farm already, the
    name of that farm will be prefixed with an asterisk in the returned list.
    """
    utils.check_for_cloud_server()
    utils.check_for_cloud_user()
    server = Server(config["cloud_server"]["url"])
    server.log_in(
        config["cloud_server"]["username"], config["cloud_server"]["password"]
    )
    farms_list = server.get_user_info().get("farms", [])
    if not len(farms_list):
        raise click.ClickException(
            "No farms exist. Run `openag cloud create_farm` to create one"
        )
    active_farm_name = config["cloud_server"]["farm_name"]
    for farm_name in farms_list:
        if farm_name == active_farm_name:
            click.echo("*"+farm_name)
        else:
            click.echo(farm_name)