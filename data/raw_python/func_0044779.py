def collapsed_dependencies(self):
        """
        Accessess collapsed dependencies for this sentence

        :getter: Returns the dependency graph for collapsed dependencies
        :type: corenlp_xml.dependencies.DependencyGraph

        """
        if self._basic_dependencies is None:
            deps = self._element.xpath('dependencies[@type="collapsed-dependencies"]')
            if len(deps) > 0:
                self._basic_dependencies = DependencyGraph(deps[0])
        return self._basic_dependencies