def init_graph(self):
        """
        Initialize graph
        Load all nodes and set dependencies.

        To avoid errors about missing nodes all nodes get loaded first before
        setting the dependencies.
        """
        self._graph = Graph()

        # First add all nodes
        for key in self.loader.disk_fixtures.keys():
            self.graph.add_node(key)

        # Then set dependencies
        for key, fixture in self.loader.disk_fixtures.items():
            for dependency in fixture.dependencies:
                self.graph.add_dependency(key, dependency)