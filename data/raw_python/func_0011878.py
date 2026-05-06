def clean(deltox=False):
    '''Delete temporary files not under version control.

    Args:
        deltox: If True, delete virtual environments used by tox
    '''

    basedir = dirname(__file__)

    print(cyan('delete temp files and dirs for packaging'))
    local(flo(
        'rm -rf  '
        '{basedir}/.eggs/  '
        '{basedir}/utlz.egg-info/  '
        '{basedir}/dist  '
        '{basedir}/README  '
        '{basedir}/build/  '
    ))

    print(cyan('\ndelete temp files and dirs for editing'))
    local(flo(
        'rm -rf  '
        '{basedir}/.cache  '
        '{basedir}/.ropeproject  '
    ))

    print(cyan('\ndelete bytecode compiled versions of the python src'))
    # cf. http://stackoverflow.com/a/30659970
    local(flo('find  {basedir}/utlz  {basedir}/tests  ') +
          '\( -name \*pyc -o -name \*.pyo -o -name __pycache__ '
          '-o -name \*.so -o -name \*.o -o -name \*.c \) '
          '-prune '
          '-exec rm -rf {} +')

    if deltox:
        print(cyan('\ndelete tox virual environments'))
        local(flo('cd {basedir}  &&  rm -rf .tox/'))