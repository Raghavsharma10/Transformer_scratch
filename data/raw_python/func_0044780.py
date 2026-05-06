def collapsed_ccprocessed_dependencies(self):
        """
        Accesses collapsed, CC-processed dependencies

        :getter: Returns the dependency graph for collapsed and cc processed dependencies
        :type: corenlp_xml.dependencies.DependencyGraph

        """
        if self._basic_dependencies is None:
            deps = self._element.xpath('dependencies[@type="collapsed-ccprocessed-dependencies"]')
            if len(deps) > 0:
                self._basic_dependencies = DependencyGraph(deps[0])
        return self._basic_dependencies