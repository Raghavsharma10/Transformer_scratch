def fix_path(p):
    """
    Convert path pointing subdirectory of virtualenv site-packages
    to system site-packages.

    Destination directory must exist for this to work.

    >>> fix_path('C:\\some-venv\\Lib\\site-packages\\gnome')
    'C:\\Python27\\lib\\site-packages\\gnome'
    """
    venv_lib = get_python_lib()

    if p.startswith(venv_lib):
        subdir = p[len(venv_lib) + 1:]

        for sitedir in getsyssitepackages():
            fixed_path = join(sitedir, subdir)
            if isdir(fixed_path):
                return fixed_path

    return p