def clusterstatus(self):
        """
        Returns a slightly slimmed down version of the clusterstatus api command. It also gets count of documents in each shard on each replica and returns
        it as doc_count key for each replica.

        """

        res = self.cluster_status_raw()

        cluster = res['cluster']['collections']
        out = {}
        try:
            for collection in cluster:
                out[collection] = {}
                for shard in cluster[collection]['shards']:
                    out[collection][shard] = {}
                    for replica in cluster[collection]['shards'][shard]['replicas']:
                        out[collection][shard][replica] = cluster[collection]['shards'][shard]['replicas'][replica]
                        if out[collection][shard][replica]['state'] != 'active':
                            out[collection][shard][replica]['doc_count'] = False
                        else:
                            out[collection][shard][replica]['doc_count'] = self._get_collection_counts(
                                out[collection][shard][replica])
        except Exception as e:
            self.logger.error("Couldn't parse response from clusterstatus API call")
            self.logger.exception(e)

        return out