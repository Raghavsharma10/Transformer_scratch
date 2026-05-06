def bundle_lambda(zipfile):
    """Write zipfile contents to file.

    :param zipfile:
    :return: exit_code
    """
    # TODO have 'bundle.zip' as default config
    if not zipfile:
        return 1
    with open('bundle.zip', 'wb') as zfile:
        zfile.write(zipfile)
    log.info('Finished - a bundle.zip is waiting for you...')
    return 0