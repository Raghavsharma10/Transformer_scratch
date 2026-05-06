def dependent(self, dep_type, node):
        """
        Registers a node as dependent on this node

        :param dep_type: The dependency type
        :type dep_type: str
        :param node: The node to be registered as a dependent
        :type node: corenlp_xml.dependencies.DependencyNode

        :return: self, provides fluent interface
        :rtype: corenlp_xml.dependencies.DependencyNode

        """
        self._dependents[dep_type] = self._dependents.get(dep_type, []) + [node]
        return self