def get_all_paths_from(self, start, seen=None):
        '''
        Return a list of all paths to all nodes from a given start node
        '''
        if seen is None:
            seen = frozenset()
        results = [(0, (start, ))]
        if start in seen or start not in self.edges:
            return results
        seen = seen | frozenset((start,))
        for node, edge_weight in self.edges[start].items():
            for subpath_weight, subpath in self.get_all_paths_from(node, seen):
                total_weight = edge_weight + subpath_weight
                full_path = (start, ) + subpath
                results.append((total_weight, full_path))
        return tuple(results)