def rollback(awsclient, function_name, alias_name=ALIAS_NAME, version=None):
    """Rollback a lambda function to a given version.

    :param awsclient:
    :param function_name:
    :param alias_name:
    :param version:
    :return: exit_code
    """
    if version:
        log.info('rolling back to version {}'.format(version))
    else:
        log.info('rolling back to previous version')
        version = _get_previous_version(awsclient, function_name, alias_name)
        if version == '0':
            log.error('unable to find previous version of lambda function')
            return 1

        log.info('new version is %s' % str(version))

    _update_alias(awsclient, function_name, version, alias_name)
    return 0