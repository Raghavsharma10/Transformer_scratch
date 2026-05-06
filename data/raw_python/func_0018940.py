def select_modeltypes(self, *models: ModelTypesArg) -> 'Selection':
        """Restrict the current |Selection| object to all elements
        containing the given model types (removes all nodes).

        See the documentation on method |Selection.search_modeltypes| for
        additional information.
        """
        self.nodes = devicetools.Nodes()
        self.elements = self.search_modeltypes(*models).elements
        return self