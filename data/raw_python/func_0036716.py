def cluster_health_for_shards(self, index=None, params={}, **kwargs):
        """
        Return a list of cluster health of specified indices(default all) and
        append shards information of each index
        the first element is a dictionary represent a global information of the cluster
        the second element represent a information of indices and its shards and each element is a dictionary
        such as [{'index' : 'a', 'status' : 'yellow', ..., 'shards' : {'0' : {...}, '1' : {...}, ...}, ...]
        """
        params['level'] = 'shards'
        result = self.cluster_health(index, params, **kwargs)
        return self._process_cluster_health_info(result)