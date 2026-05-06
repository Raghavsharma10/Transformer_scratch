def governor(self, dep_type, node):
        """
        Registers a node as governing this node

        :param dep_type: The dependency type
        :type dep_type: str
        :param node:

        :return: self, provides fluent interface
        :rtype: corenlp_xml.dependencies.DependencyNode

        """
        self._governors[dep_type] = self._governors.get(dep_type, []) + [node]
        return self