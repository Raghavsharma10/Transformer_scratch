def connected_components_subgraphs(self, copy=True):
        """ Iterates over connected components in current :class:`BreakpointGraph` object, and yields new instances of :class:`BreakpointGraph` with respective information deep-copied by default (week reference is possible of specified in method call).

        :param copy: a flag to signal if graph information has to be deep copied while producing new :class:`BreakpointGraph` instances, of just reference to respective data has to be made.
        :type copy: ``Boolean``
        :return: generator over connected components in current :class:`BreakpointGraph` wrapping respective connected components into new :class:`BreakpointGraph` objects.
        :rtype: ``generator``
        """
        for component in nx.connected_components(self.bg):
            component = self.bg.subgraph(component)
            if copy:
                component.copy()
            yield BreakpointGraph(component)