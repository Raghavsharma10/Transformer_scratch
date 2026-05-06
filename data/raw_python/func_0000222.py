def find_executable(name: str, flags=os.X_OK) -> List[str]:
    r"""Finds executable `name`.

    Similar to Unix ``which`` command.

    Returns list of zero or more full paths to `name`.
    """
    result = []
    extensions = [x for x in os.environ.get("PATHEXT", "").split(os.pathsep) if x]
    path = os.environ.get("PATH", None)
    if path is None:
        return []
    for path in os.environ.get("PATH", "").split(os.pathsep):
        path = os.path.join(path, name)
        if os.access(path, flags):
            result.append(path)
        for extension in extensions:
            path_extension = path + extension
            if os.access(path_extension, flags):
                result.append(path_extension)
    return result