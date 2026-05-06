def define(self, schema, *, validate=True):
        """ Store the task graph definition (schema).

        The schema has to adhere to the following rules:

        A key in the schema dict represents a parent task and the value one or more
        children:
            {parent: [child]} or {parent: [child1, child2]}

        The data output of one task can be routed to a labelled input slot of successor
        tasks using a dictionary instead of a list for the children:
            {parent: {child1: 'positive', child2: 'negative'}}

        An empty slot name or None skips the creation of a labelled slot:
            {parent: {child1: '', child2: None}}

        Args:
            schema (dict): A dictionary with the schema definition.
            validate (bool): Set to True to validate the graph by checking whether it is
                             a directed acyclic graph.
        """
        self._schema = schema
        if validate:
            self.validate(self.make_graph(self._schema))