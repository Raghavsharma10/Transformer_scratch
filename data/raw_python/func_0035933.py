def chmod(path, mode, operator='=', recursive=False):
    '''
    Change file mode bits.

    When recursively chmodding a directory, executable bits in ``mode`` are
    ignored when applying to a regular file. E.g. ``chmod(path, mode=0o777,
    recursive=True)`` would apply ``mode=0o666`` to regular files.

    Symlinks are ignored.

    Parameters
    ----------
    path : ~pathlib.Path
        Path to chmod.
    mode : int
        Mode bits to apply, e.g. ``0o777``.
    operator : str
        How to apply the mode bits to the file, one of:

        '='
            Replace mode with given mode.
        '+'
            Add to current mode.
        '-'
            Subtract from current mode.

    recursive : bool
        Whether to chmod recursively.
    '''
    if mode > 0o777 and operator != '=':
        raise ValueError('Special bits (i.e. >0o777) only supported when using "=" operator')

    # first chmod path
    if operator == '+':
        mode_ = path.stat().st_mode | mode
    elif operator == '-':
        mode_ = path.stat().st_mode & ~mode
    else:
        mode_ = mode
    if path.is_symlink():
        # Do not chmod or follow symlinks
        return
    path.chmod(mode_)

    # then its children
    def chmod_children(parent, files, mode_mask, operator):
        for file in files:
            with suppress(FileNotFoundError):
                file = parent / file
                if not file.is_symlink():
                    chmod(file, mode & mode_mask, operator)
    if recursive and path.is_dir():
        for parent, dirs, files in os.walk(str(path)):
            parent = Path(parent)
            chmod_children(parent, dirs, 0o777777, operator)
            chmod_children(parent, files, 0o777666, operator)