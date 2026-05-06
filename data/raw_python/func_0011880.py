def tox(args=''):
    '''Run tox.

    Build package and run unit tests against several pythons.

    Args:
        args: Optional arguments passed to tox.
        Example:

            fab tox:'-e py36 -r'
    '''
    basedir = dirname(__file__)

    latest_pythons = _determine_latest_pythons()
    # e.g. highest_minor_python: '3.6'
    highest_minor_python = _highest_minor(latest_pythons)

    _local_needs_pythons(flo('cd {basedir}  &&  '
                             'python{highest_minor_python} -m tox {args}'))