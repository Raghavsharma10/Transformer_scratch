def _submit_changes_to_github_repo(path, url):
    """ Temporarily commits local changes and submits them to
    the GitHub repository that the user has specified. Then
    reverts the changes to the git repository if a commit was
    necessary. """
    try:
        repo = git.Repo(path)
    except Exception:
        raise RuntimeError('Couldn\'t locate a repository at `%s`.' % path)
    commited = False
    try:
        try:
            repo.delete_remote('trytravis')
        except Exception:
            pass
        print('Adding a temporary remote to '
              '`%s`...' % url)
        remote = repo.create_remote('trytravis', url)

        print('Adding all local changes...')
        repo.git.add('--all')
        try:
            print('Committing local changes...')
            timestamp = datetime.datetime.now().isoformat()
            repo.git.commit(m='trytravis-' + timestamp)
            commited = True
        except git.exc.GitCommandError as e:
            if 'nothing to commit' in str(e):
                commited = False
            else:
                raise
        commit = repo.head.commit.hexsha
        committed_at = repo.head.commit.committed_datetime

        print('Pushing to `trytravis` remote...')
        remote.push(force=True)
    finally:
        if commited:
            print('Reverting to old state...')
            repo.git.reset('HEAD^')
        try:
            repo.delete_remote('trytravis')
        except Exception:
            pass
    return commit, committed_at