def create(name, packages=None, user=None):
    """
    Create a conda env
    """
    packages = packages or ''
    packages = packages.split(',')
    packages.append('pip')
    args = packages + ['--yes', '-q']
    cmd = _create_conda_cmd('create', args=args, env=name, user=user)
    ret = _execcmd(cmd, user=user, return0=True)

    if ret['retcode'] == 0:
        ret['result'] = True
        ret['comment'] = 'Virtual enviroment "%s" successfully created' % name
    else:
        if ret['stderr'].startswith('Error: prefix already exists:'):
            ret['result'] = True
            ret['comment'] = 'Virtual enviroment "%s" already exists' % name
        else:
            ret['result'] = False
            ret['error'] = salt.exceptions.CommandExecutionError(ret['stderr'])
    return ret