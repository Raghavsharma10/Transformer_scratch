def _pypi_get_projects_for_user(username):
    """
    Given the username of a PyPI user, return a list of all of the user's
    projects from the XMLRPC interface.

    See: https://wiki.python.org/moin/PyPIXmlRpc

    :param username: PyPI username
    :type username: str
    :return: list of string project names
    :rtype: ``list``
    """
    client = xmlrpclib.ServerProxy('https://pypi.python.org/pypi')
    pkgs = client.user_packages(username)  # returns [role, package]
    return [x[1] for x in pkgs]