def reindex(self, fq= [], **kwargs):
        '''
        Starts Reindexing Process. All parameter arguments will be passed down to the getter function.
        :param string fq: FilterQuery to pass to source Solr to retrieve items. This can be used to limit the results.
        '''
        for items in self._getter(fq=fq, **kwargs):
            self._putter(items)
        if type(self._dest) is SolrClient and self._dest_coll:
            self.log.info("Finished Indexing, sending a commit")
            self._dest.commit(self._dest_coll, openSearcher=True)