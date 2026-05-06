def register(username, password):
    """
    Create a new user account. Creates a user account with the given
    credentials on the selected cloud server.
    """
    check_for_cloud_server()
    server = Server(config["cloud_server"]["url"])
    server.create_user(username, password)