def get_parent_log_nodes(self):
        """Gets the parents of this log.

        return: (osid.logging.LogNodeList) - the parents of this log
        *compliance: mandatory -- This method must be implemented.*

        """
        parent_log_nodes = []
        for node in self._my_map['parentNodes']:
            parent_log_nodes.append(LogNode(
                node._my_map,
                runtime=self._runtime,
                proxy=self._proxy,
                lookup_session=self._lookup_session))
        return LogNodeList(parent_log_nodes)