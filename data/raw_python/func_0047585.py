def read_tgf(filename):
    """Reads a file in Trivial Graph Format."""
    g = Graph()
    with open(filename) as file:
        states = {}

        # Nodes
        for line in file:
            line = line.strip()
            if line == "": 
                continue
            elif line == "#":
                break
            i, q = line.split(None, 1)
            q, attrs = syntax.string_to_state(q)
            states[i] = q
            g.add_node(q, attrs)

        # Edges
        for line in file:
            line = line.strip()
            if line == "": 
                continue
            i, j, t = line.split(None, 2)
            q, r = states[i], states[j]
            t = syntax.string_to_transition(t)
            g.add_edge(q, r, {'label':t})

    return from_graph(g)