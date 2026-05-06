def get_processes(sort_by_name=True):
    """Retrieve a list of processes sorted by name.

    Args:
        sort_by_name (bool): Sort the list by name or by process ID's.

    Returns:
        list of (int, str) or list of (int, str, str): List of process id,
            process name and optional cmdline tuples.
    """
    if sort_by_name:
        return sorted(
            _list_processes(),
            key=cmp_to_key(
                lambda p1, p2: (cmp(p1.name, p2.name) or cmp(p1.pid, p2.pid))
            ),
        )
    else:
        return sorted(
            _list_processes(),
            key=cmp_to_key(
                lambda p1, p2: (cmp(p1.pid, p2.pid) or cmp(p1.name, p2.name))
            ),
        )