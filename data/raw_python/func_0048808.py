def login(username, password):
    """ Log into your user account """
    check_for_cloud_server()
    old_username = config["cloud_server"]["username"]
    if old_username and old_username != username:
        raise click.ClickException(
            "Already logged in as user \"{}\". Run `openag cloud user logout` "
            "before attempting to log in as a different user".format(
                old_username
            )
        )
    server = Server(config["cloud_server"]["url"])
    server.log_in(username, password)
    config["cloud_server"]["username"] = username
    config["cloud_server"]["password"] = password