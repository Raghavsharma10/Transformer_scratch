def draw(args):
    """Load the build graph for a target and render it to an image."""
    if len(args) != 2:
        log.error('Two arguments required: [build target] [output file]')
        app.quit(1)

    target = args[0]
    out = args[1]

    try:
        bb = Butcher()
        bb.load_graph(target)
    except error.BrokenGraph as lolno:
        log.fatal(lolno)
        app.quit(1)

    # Filter down to the target and all of its transitive dependencies.
    # TODO: make it possible to optionally draw the entire graph
    filtered_graph = bb.graph.subgraph(
        networkx.topological_sort(bb.graph, nbunch=[address.new(target)]))

    a = networkx.to_agraph(filtered_graph)
    a.draw(out, prog='dot')
    log.info('Graph written to %s', out)