def are_credentials_still_valid(awsclient):
    """Check whether the credentials have expired.

    :param awsclient:
    :return: exit_code
    """
    client = awsclient.get_client('lambda')
    try:
        client.list_functions()
    except GracefulExit:
        raise
    except Exception as e:
        log.debug(e)
        log.error(e)
        return 1
    return 0