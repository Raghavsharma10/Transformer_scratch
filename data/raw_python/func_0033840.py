def find_server():
    """Find the default server host from the environment

    This method uses the C{LIGO_DATAFIND_SERVER} variable to construct
    a C{(host, port)} tuple.

    @returns: C{(host, port)}: the L{str} host name and L{int} port number

    @raises RuntimeError: if the C{LIGO_DATAFIND_SERVER} environment variable
                          is not set
    """

    if os.environ.has_key(_server_env):
        host = os.environ[_server_env]
        port = None
        if re.search(':', host):
            host, port = host.split(':', 1)
            if port:
                port = int(port)
        return host, port
    else:
        raise RuntimeError("Environment variable %s is not set" % _server_env)