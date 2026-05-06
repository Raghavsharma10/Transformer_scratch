def login(username=None, password=None):
    """
    Log in to PNC using the supplied username and password. The keycloak token will
    be saved for all subsequent pnc-cli operations until login is called again
    :return:
    """
    global user
    user = UserConfig()
    if username:
        user.username = username
    else:
        user.username = user.input_username()

    if password:
        user.password = password
    else:
        user.password = user.input_password()

    if (not ( user.username and user.password) ):
        logging.error("Username and password must be provided for login")
        return;
    user.retrieve_keycloak_token()
    user.apiclient = user.create_api_client()
    save()