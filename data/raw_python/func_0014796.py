def get_ready_data_nodes(self, seed_path, gather_depth):
        """Returns a list [(path1,data_node1),...]
        with entries only for existing nodes with DataObjects where is_ready==True.
        Missing nodes or those with non-ready or non-existing data are ignored.
        """
        try:
            seed_node = self.get_node(seed_path)
        except MissingBranchError:
            return []
        all_paths = seed_node._get_all_paths(seed_path, gather_depth)
        ready_data_nodes = []
        for path in all_paths:
            if self.is_ready(data_path=path):
                ready_data_nodes.append((path, self.get_node(path)))
        return ready_data_nodes