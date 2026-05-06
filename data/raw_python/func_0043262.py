def remove(packages, env=None, user=None):
    """
    Remove conda packages in a conda env

    Attributes
    ----------
        packages: list of packages comma delimited
    """
    packages = ' '.join(packages.split(','))
    cmd = _create_conda_cmd('remove', args=[packages, '--yes', '-q'], env=env, user=user)
    return _execcmd(cmd, user=user, return0=True)