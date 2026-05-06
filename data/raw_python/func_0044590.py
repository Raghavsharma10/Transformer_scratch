def dependent(self):
        """
        Accesses the dependent node

        :getter: returns the Dependent node
        :type: corenlp_xml.dependencies.DependencyNode

        """
        if self._dependent is None:
            dependents = self._element.xpath('dependent')
            if len(dependents) > 0:
                self._dependent = DependencyNode.load(self._graph, dependents[0])
        return self._dependent