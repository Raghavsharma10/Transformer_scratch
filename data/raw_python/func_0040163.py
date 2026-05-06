def _clean():
    """
    Cleans up build dir
    """
    LOGGER.info('Cleaning project directory...')
    folders_to_cleanup = [
        '.eggs',
        'build',
        f'{config.PACKAGE_NAME()}.egg-info',
    ]
    for folder in folders_to_cleanup:
        if os.path.exists(folder):
            LOGGER.info('\tremoving: %s', folder)
            shutil.rmtree(folder)