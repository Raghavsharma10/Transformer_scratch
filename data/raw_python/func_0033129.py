def _map_filtered_clusters_to_full_clusters(self,
                                                clusters,
                                                filter_map):
        """
            Input:  clusters, a list of cluster lists
                    filter_map, the seq_id in each clusters
                                is the key to the filter_map
                                containing all seq_ids with
                                duplicate FASTA sequences
            Output: an extended list of cluster lists
        """
        results = []
        for cluster in clusters:
            full_cluster = []
            for seq_id in cluster:
                full_cluster += filter_map[seq_id]
            results.append(full_cluster)
        return results