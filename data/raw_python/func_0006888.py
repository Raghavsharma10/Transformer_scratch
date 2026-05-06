def _split_docker_link(alias_name):
    """
    Splits a docker link string into a list of 3 items (protocol, host, port).
    - Assumes IPv4 Docker links

    ex: _split_docker_link('DB') -> ['tcp', '172.17.0.82', '8080']
    """
    sanitized_name = alias_name.strip().upper()
    split_list = re.split(r':|//', core.str('{0}_PORT'.format(sanitized_name)))
    # filter out empty '' vals from the list with filter and
    # cast to list (required for python3)
    return list(filter(None, split_list))