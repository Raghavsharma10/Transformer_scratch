def cli(context, host, username, password):
    """
    FritzBox SmartHome Tool

    \b
    Provides the following functions:
    - A easy to use library for querying SmartHome actors
    - This CLI tool for testing
    - A carbon client for pipeing data into graphite
    """
    context.obj = FritzBox(host, username, password)