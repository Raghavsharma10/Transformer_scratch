def _find_proj_root():
    # type: () -> Optional[str]
    """ Find the project path by going up the file tree.

    This will look in the current directory and upwards for the pelconf file
    (.yaml or .py)
    """
    proj_files = frozenset(('pelconf.py', 'pelconf.yaml'))
    curr = os.getcwd()

    while curr.startswith('/') and len(curr) > 1:
        if proj_files & frozenset(os.listdir(curr)):
            return curr
        else:
            curr = os.path.dirname(curr)

    return None