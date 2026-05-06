def host(value):
    """ Validates that the value is a valid network location """
    if not value:
        return (True, "")
    try:
        host,port = value.split(":")
    except ValueError as _:
        return (False, "value needs to be <host>:<port>")

    try:
        int(port)
    except ValueError as _:
        return (False, "port component of the host address needs to be a number")

    return (True, "")