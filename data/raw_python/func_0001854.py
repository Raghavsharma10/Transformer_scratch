def compare_files(path1, path2):
    # type: (str, str) -> List[str]
    """Returns the delta between two files using -, ?, + format excluding
    lines that are the same

    Args:
        path1 (str): Path to first file
        path2 (str): Path to second file

    Returns:
        List[str]: Delta between the two files

    """
    diff = difflib.ndiff(open(path1).readlines(), open(path2).readlines())
    return [x for x in diff if x[0] in ['-', '+', '?']]