def basic_dependencies(self):
        """
        Accesses basic dependencies from the XML output

        :getter: Returns the dependency graph for basic dependencies
        :type: corenlp_xml.dependencies.DependencyGraph

        """
        if self._basic_dependencies is None:
            deps = self._element.xpath('dependencies[@type="basic-dependencies"]')
            if len(deps) > 0:
                self._basic_dependencies = DependencyGraph(deps[0])
        return self._basic_dependencies