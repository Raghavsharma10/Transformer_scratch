def _git_receive_v1(msg, tmpl, **config):
    ''' Return the subtitle for the first version of pagure git.receive
    messages.
    '''
    repo = _get_project(msg['msg']['commit'], key='repo')
    email = msg['msg']['commit']['email']
    user = email2fas(email, **config)
    summ = msg['msg']['commit']['summary']
    whole = msg['msg']['commit']['message']
    if summ.strip() != whole.strip():
        summ += " (..more)"

    branch = msg['msg']['commit']['branch']
    if 'refs/heads/' in branch:
        branch = branch.replace('refs/heads/', '')
    return tmpl.format(user=user or email, repo=repo,
                       branch=branch, summary=summ)