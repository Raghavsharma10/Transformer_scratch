def _load_github_repo():
    """ Loads the GitHub repository from the users config. """
    if 'TRAVIS' in os.environ:
        raise RuntimeError('Detected that we are running in Travis. '
                           'Stopping to prevent infinite loops.')
    try:
        with open(os.path.join(config_dir, 'repo'), 'r') as f:
            return f.read()
    except (OSError, IOError):
        raise RuntimeError('Could not find your repository. '
                           'Have you ran `trytravis --repo`?')