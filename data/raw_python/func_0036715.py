def cluster_health_for_indices(self, index=None, params={}, **kwargs):
        """
        Return a list of cluster health of specified indices(default all),
        the first element is a dictionary represent a global information of the cluster
        such as "cluster_name", "number_of_nodes"...
        the second element represent a indices information list that each element is a dictionary for one index
        such as [{'index' : 'a', 'status' : 'yellow', ...} , {'index' : 'b', 'status' : 'yellow', ...}, ....]
        """
        params['level'] = 'indices'
        result = self.cluster_health(index, params, **kwargs)
        return self._process_cluster_health_info(result)