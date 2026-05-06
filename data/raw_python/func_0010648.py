def make_graph(schema):
        """ Construct the task graph (dag) from a given schema.

        Parses the graph schema definition and creates the task graph. Tasks are the
        vertices of the graph and the connections defined in the schema become the edges.

        A key in the schema dict represents a parent task and the value one or more
        children:
            {parent: [child]} or {parent: [child1, child2]}

        The data output of one task can be routed to a labelled input slot of successor
        tasks using a dictionary instead of a list for the children:
            {parent: {child1: 'positive', child2: 'negative'}}

        An empty slot name or None skips the creation of a labelled slot:
            {parent: {child1: '', child2: None}}

        The underlying graph library creates nodes automatically, when an edge between
        non-existing nodes is created.

        Args:
            schema (dict): A dictionary with the schema definition.

        Returns:
            DiGraph: A reference to the fully constructed graph object.

        Raises:
            DirectedAcyclicGraphUndefined: If the schema is not defined.
        """
        if schema is None:
            raise DirectedAcyclicGraphUndefined()

        # sanitize the input schema such that it follows the structure:
        #    {parent: {child_1: slot_1, child_2: slot_2, ...}, ...}
        sanitized_schema = {}
        for parent, children in schema.items():
            child_dict = {}
            if children is not None:
                if isinstance(children, list):
                    if len(children) > 0:
                        child_dict = {child: None for child in children}
                    else:
                        child_dict = {None: None}
                elif isinstance(children, dict):
                    for child, slot in children.items():
                        child_dict[child] = slot if slot != '' else None
                else:
                    child_dict = {children: None}
            else:
                child_dict = {None: None}

            sanitized_schema[parent] = child_dict

        # build the graph from the sanitized schema
        graph = nx.DiGraph()
        for parent, children in sanitized_schema.items():
            for child, slot in children.items():
                if child is not None:
                    graph.add_edge(parent, child, slot=slot)
                else:
                    graph.add_node(parent)

        return graph