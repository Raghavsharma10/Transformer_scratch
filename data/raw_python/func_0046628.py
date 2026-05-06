def merge_commit(commit):
    "Fetches the latest code and merges up the specified commit."
    with cd(env.path):
        run('git fetch')
        if '@' in commit:
            branch, commit = commit.split('@')
            run('git checkout {0}'.format(branch))
        run('git merge {0}'.format(commit))