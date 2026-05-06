def pypi():
    '''Build package and upload to pypi.'''
    if query_yes_no('version updated in setup.py?'):

        print(cyan('\n## clean-up\n'))
        execute(clean)

        basedir = dirname(__file__)

        latest_pythons = _determine_latest_pythons()
        # e.g. highest_minor: '3.6'
        highest_minor = _highest_minor(latest_pythons)
        python = flo('python{highest_minor}')

        print(cyan('\n## build package'))
        _local_needs_pythons(flo('cd {basedir}  &&  {python}  setup.py  sdist'))

        print(cyan('\n## upload package'))
        local(flo('cd {basedir}  &&  {python} -m twine upload  dist/*'))