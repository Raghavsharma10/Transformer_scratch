def get_parent_repository_nodes(self):
        """Gets the parents of this repository.

        return: (osid.repository.RepositoryNodeList) - the parents of
                the ``id``
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_repository_nodes = []
        for node in self._my_map['parentNodes']:
            parent_repository_nodes.append(RepositoryNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return RepositoryNodeList(parent_repository_nodes)