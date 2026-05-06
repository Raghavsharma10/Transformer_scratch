def delete_dir_rec(path: Path):
    """
    Delete a folder recursive.

    :param path: folder to deleted
    :type path: ~pathlib.Path
    """
    if not path.exists() or not path.is_dir():
        return
    for sub in path.iterdir():
        if sub.is_dir():
            delete_dir_rec(sub)
        else:
            sub.unlink()
    path.rmdir()