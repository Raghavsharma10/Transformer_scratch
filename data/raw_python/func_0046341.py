def create_farm(farm_name):
    """
    Create a farm. Creates a farm named FARM_NAME on the currently selected
    cloud server. You can use the `openag cloud select_farm` command to start
    mirroring data into it.
    """
    utils.check_for_cloud_server()
    utils.check_for_cloud_user()
    server = Server(config["cloud_server"]["url"])
    username = config["cloud_server"]["username"]
    password = config["cloud_server"]["password"]
    server.log_in(username, password)
    url = urljoin(server.resource.url, "_openag", "v0.1", "register_farm")
    status, _, content = server.resource.session.request(
        "POST", url, headers=server.resource.headers.copy(), body={
            "name": username, "farm_name": farm_name
        }, credentials=(username, password)
    )
    if status != 200:
        raise click.ClickException(
            "Failed to register farm with cloud server ({}): {}".format(
                status, content
            )
        )