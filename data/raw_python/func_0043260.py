def list_(env=None, user=None):
    """
    List the installed packages on an environment

    Returns
    -------
        Dictionary: {package: {version: 1.0.0, build: 1 } ... }
    """
    cmd = _create_conda_cmd('list', args=['--json'], env=env, user=user)
    ret = _execcmd(cmd, user=user)
    if ret['retcode'] == 0:
        pkg_list = json.loads(ret['stdout'])
        packages = {}
        for pkg in pkg_list:
            pkg_info = pkg.split('-')
            name, version, build = '-'.join(pkg_info[:-2]), pkg_info[-2], pkg_info[-1]
            packages[name] = {'version': version, 'build': build}
        return packages
    else:
        return ret