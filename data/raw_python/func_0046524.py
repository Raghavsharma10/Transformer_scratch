def show():
    """
    Shows the URL of the current cloud server or throws an error if no cloud
    server is selected
    """
    utils.check_for_cloud_server()
    click.echo("Using cloud server at \"{}\"".format(
        config["cloud_server"]["url"]
    ))
    if config["cloud_server"]["username"]:
        click.echo(
            "Logged in as user \"{}\"".format(config["cloud_server"]["username"])
        )
    if config["cloud_server"]["farm_name"]:
        click.echo(
            "Using farm \"{}\"".format(config["cloud_server"]["farm_name"])
        )