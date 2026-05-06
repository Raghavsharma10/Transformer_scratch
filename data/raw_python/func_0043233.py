def managed(name, packages=None, requirements=None, saltenv='base', user=None):
    """
    Create and install python requirements in a conda enviroment
    pip is installed by default in the new enviroment

    name : path to the enviroment to be created
    packages : None
        single package or list of packages to install i.e. numpy, scipy=0.13.3, pandas
    requirements : None
        path to a `requirements.txt` file in the `pip freeze` format
    saltenv : 'base'
        Salt environment. Usefull when the name is file using the salt file system
        (e.g. `salt://.../reqs.txt`)
    user
        The user under which to run the commands
    """
    ret = {'name': name, 'changes': {}, 'comment': '', 'result': True}
    comments = []

    # Create virutalenv
    try:
        installation_comment = __salt__['conda.create'](name, user=user)
        if installation_comment.endswith('created'):
            comments.append('Virtual enviroment "%s" created' % name)
        else:
            comments.append('Virtual enviroment "%s" already exists' % name)
    except Exception as e:
        ret['comment'] = e
        ret['result'] = False
        return ret

    # Install packages
    if packages is not None:
        installation_ret = installed(packages, env=name, saltenv=saltenv, user=user)
        ret['result'] = ret['result'] and installation_ret['result']
        comments.append('From list [%s]' % installation_ret['comment'])
        ret['changes'].update(installation_ret['changes'])

    if requirements is not None:
        installation_ret = installed(requirements, env=name, saltenv=saltenv, user=user)
        ret['result'] = ret['result'] and installation_ret['result']
        comments.append('From file [%s]' % installation_ret['comment'])
        ret['changes'].update(installation_ret['changes'])

    ret['comment'] = '. '.join(comments)
    return ret