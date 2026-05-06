def getContainerStats(self, limit=None, marker=None):
        """Returns Rackspace Cloud Files usage stats for containers.
        
        @param limit:  Number of containers to return.
        @param marker: Return only results whose name is greater than marker.
        @return:       Dictionary of container stats indexed by container name.
        
        """
        stats = {}
        for row in self._conn.list_containers_info(limit, marker):
            stats[row['name']] = {'count': row['count'], 'size': row['bytes']}
        return stats