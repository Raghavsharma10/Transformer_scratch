def optimize_layout_handler(rule, handler):
    """
    Produce an "optimized" version of handler for the dispatcher to
    limit reference lookups.
    """

    def runner(walk, dispatcher, node):
        yield LayoutChunk(rule, handler, node)

    return runner