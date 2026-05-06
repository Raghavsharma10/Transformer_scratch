def write(
    contents: str,
    path: Union[str, pathlib.Path],
    verbose: bool = False,
    logger_func=None,
) -> bool:
    """
    Writes ``contents`` to ``path``.

    Checks if ``path`` already exists and only write out new contents if the
    old contents do not match.

    Creates any intermediate missing directories.

    :param contents: the file contents to write
    :param path: the path to write to
    :param verbose: whether to print output
    """
    print_func = logger_func or print
    path = pathlib.Path(path)
    if path.exists():
        with path.open("r") as file_pointer:
            old_contents = file_pointer.read()
        if old_contents == contents:
            if verbose:
                print_func("preserved {}".format(path))
            return False
        else:
            with path.open("w") as file_pointer:
                file_pointer.write(contents)
            if verbose:
                print_func("rewrote {}".format(path))
            return True
    elif not path.exists():
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with path.open("w") as file_pointer:
            file_pointer.write(contents)
        if verbose:
            print_func("wrote {}".format(path))
    return True