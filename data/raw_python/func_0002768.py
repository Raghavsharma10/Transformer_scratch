def dir_exists(directory):
    """
    If a directory already exists that will be overwritten by some action, this
    will ask the user whether or not to continue with the deletion.

    If the user responds affirmatively, then the directory will be removed. If
    the user responds negatively, then the process will abort.
    """
    log.info('Directory exists! Asking the user')
    reply = input('''The directory {0} already exists.
It will be overwritten if the operation continues.
Replace? [Y/n]'''.format(directory))
    if reply.lower() in ['y', 'yes', '']:
        shutil.rmtree(directory)
        os.makedirs(directory)
    else:
        log.critical('Aborting process, user declined overwriting {0}'.format(directory))
        sys.exit('Aborting process!')