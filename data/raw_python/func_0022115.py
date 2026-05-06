def _input_github_repo(url=None):
    """ Grabs input from the user and saves
    it as their trytravis target repo """
    if url is None:
        url = user_input('Input the URL of the GitHub repository '
                         'to use as a `trytravis` repository: ')
    url = url.strip()
    http_match = _HTTPS_REGEX.match(url)
    ssh_match = _SSH_REGEX.match(url)
    if not http_match and not ssh_match:
        raise RuntimeError('That URL doesn\'t look like a valid '
                           'GitHub URL. We expect something '
                           'of the form: `https://github.com/[USERNAME]/'
                           '[REPOSITORY]` or `ssh://git@github.com/'
                           '[USERNAME]/[REPOSITORY]')

    # Make sure that the user actually made a new repository on GitHub.
    if http_match:
        _, name = http_match.groups()
    else:
        _, name = ssh_match.groups()
    if 'trytravis' not in name:
        raise RuntimeError('You must have `trytravis` in the name of your '
                           'repository. This is a security feature to reduce '
                           'chances of running git push -f on a repository '
                           'you don\'t mean to.')

    # Make sure that the user actually wants to use this repository.
    accept = user_input('Remember that `trytravis` will make commits on your '
                        'behalf to `%s`. Are you sure you wish to use this '
                        'repository? Type `y` or `yes` to accept: ' % url)
    if accept.lower() not in ['y', 'yes']:
        raise RuntimeError('Operation aborted by user.')

    if not os.path.isdir(config_dir):
        os.makedirs(config_dir)
    with open(os.path.join(config_dir, 'repo'), 'w+') as f:
        f.truncate()
        f.write(url)
    print('Repository saved successfully.')