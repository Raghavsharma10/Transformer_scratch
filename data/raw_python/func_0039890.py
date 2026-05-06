def credentials(login=None):
    """
    Find user credentials. We should have parsed the command line for a ``--login`` option.
    We will try to find credentials in environment variables.
    We will ask user if we cannot find any in arguments nor environment
    """
    if not login:
        login = environ.get("PROF_LOGIN")
    password = environ.get("PROF_PASSWORD")
    if not login:
        try:
            login = input("login? ")
            print("\t\tDon't get prompted everytime. Store your login in the ``~/.profrc`` config file")
        except KeyboardInterrupt:
            exit(0)
    if not password:
        try:
            password = getpass.getpass("pass for {0} ? ".format(login))
        except KeyboardInterrupt:
            exit(0)
    return (login, password)