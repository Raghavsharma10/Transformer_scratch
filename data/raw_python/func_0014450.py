def _get_collection_counts(self, core_data):
        """
        Queries each core to get individual counts for each core for each shard.
        """
        if core_data['base_url'] not in self.solr_clients:
            from SolrClient import SolrClient
            self.solr_clients['base_url'] = SolrClient(core_data['base_url'], log=self.logger)
        try:
            return self.solr_clients['base_url'].query(core_data['core'],
                                                       {'q': '*:*',
                                                        'rows': 0,
                                                        'distrib': 'false',
                                                        }).get_num_found()
        except Exception as e:
            self.logger.error("Couldn't get Counts for {}/{}".format(core_data['base_url'], core_data['core']))
            self.logger.exception(e)
            return False