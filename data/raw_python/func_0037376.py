def _git_receive_v2(msg, tmpl):
    ''' Return the subtitle for the second version of pagure git.receive
    messages.
    '''
    repo = _get_project(msg['msg'], key='repo')
    user = msg['msg']['agent']
    n_commits = msg['msg']['total_commits']
    commit_lbl = 'commit' if str(n_commits) == '1' else 'commits'
    branch = msg['msg']['branch']
    if 'refs/heads/' in branch:
        branch = branch.replace('refs/heads/', '')
    return tmpl.format(user=user, repo=repo,
                       branch=branch, n_commits=n_commits,
                       commit_lbl=commit_lbl)