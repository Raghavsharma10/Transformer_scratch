def deploy(project, version, promote, quiet):
    """ Deploy the app to the target environment.

    The target environments can be configured using the ENVIRONMENTS conf
    variable. This will also collect all static files and compile translation
    messages
    """
    from . import logic

    logic.deploy(project, version, promote, quiet)