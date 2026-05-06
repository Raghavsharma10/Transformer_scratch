def build_dependency_graph(start_nodes, js_doc):
    """
    Build a graph where nodes are filenames and edges are reverse dependencies
    (so an edge from jquery.js to jquery.dimensions.js indicates that jquery.js
    must be included before jquery.dimensions.js).  The graph is represented
    as a dictionary from filename to (in-degree, edges) pair, for ease of
    topological sorting.  Also returns a list of nodes of degree zero.
    """
    queue = []
    dependencies = {}
    start_sort = []
    def add_vertex(file):
        in_degree = len(js_doc[file].module.dependencies)
        dependencies[file] = [in_degree, []]
        queue.append(file)
        if in_degree == 0:
            start_sort.append(file)
    def add_edge(from_file, to_file):
        dependencies[from_file][1].append(to_file)
    def is_in_graph(file):
        return file in dependencies

    for file in start_nodes:
        add_vertex(file)
    for file in queue:
        for dependency in js_doc[file].module.dependencies:
            if dependency not in js_doc:
                raise MissingDependency(file, dependency)
            if not is_in_graph(dependency):
                add_vertex(dependency)
            add_edge(dependency, file)
    return dependencies, start_sort