def stringify_with_dot_if_path(x):
    '''Pathlib never renders a leading './' in front of a local path. That's an
    issue because on POSIX subprocess.py (like bash) won't execute scripts in
    the current directory without it. In the same vein, we also don't want
    Path('echo') to match '/usr/bin/echo' from the $PATH. To work around both
    issues, we explicitly join a leading dot to any relative pathlib path.'''
    if isinstance(x, PurePath):
        # Note that join does nothing if the path is absolute.
        return os.path.join('.', str(x))
    return x