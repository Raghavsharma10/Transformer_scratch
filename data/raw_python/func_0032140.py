def graph(self):
        """ A conjunctive graph of all statements in the current instance. """
        if not hasattr(self, '_graph') or self._graph is None:
            self._graph = ConjunctiveGraph(store=self.store,
                                           identifier=self.base_uri)
        return self._graph