def governor(self):
        """
        Accesses the governor node

        :getter: Returns the Governor node
        :type: corenlp_xml.dependencies.DependencyNode

        """
        if self._governor is None:
            governors = self._element.xpath('governor')
            if len(governors) > 0:
                self._governor = DependencyNode.load(self._graph, governors[0])
        return self._governor