def item_from_topics(key, topics):
    """Get binding from `topics` via `key`

    Example:
        {0} == hello --> be in hello world
        {1} == world --> be in hello world

    Returns:
        Single topic matching the key

    Raises:
        IndexError (int): With number of required
            arguments for the key

    """

    if re.match("{\d+}", key):
        pos = int(key.strip("{}"))
        try:
            binding = topics[pos]
        except IndexError:
            raise IndexError(pos + 1)

    else:
        echo("be.yaml template key not recognised")
        sys.exit(PROJECT_ERROR)

    return binding