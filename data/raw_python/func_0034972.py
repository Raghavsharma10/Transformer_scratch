def optimize_structure_handler(rule, handler):
    """
    Produce an "optimized" version of handler for the dispatcher to
    limit reference lookups.
    """

    def runner(walk, dispatcher, node):
        handler(dispatcher, node)
        return
        yield  # pragma: no cover

    return runner