def ensure():
        """
        Makes sure the current working directory is a Git repository.
        """
        LOGGER.debug('checking repository')
        if not os.path.exists('.git'):
            LOGGER.error('This command is meant to be ran in a Git repository.')
            sys.exit(-1)
        LOGGER.debug('repository OK')