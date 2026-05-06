def index(self, solr, collection, threads=1, send_method='stream_file', **kwargs):
        '''
        Will index the queue into a specified solr instance and collection. Specify multiple threads to make this faster, however keep in mind that if you specify multiple threads the items may not be in order.
        Example::
            solr = SolrClient('http://localhost:8983/solr/')
            for doc in self.docs:
                index.add(doc, finalize=True)
            index.index(solr,'SolrClient_unittest')

        :param object solr: SolrClient object.
        :param string collection: The name of the collection to index document into.
        :param int threads: Number of simultaneous threads to spin up for indexing.
        :param string send_method: SolrClient method to execute for indexing. Default is stream_file
        '''

        try:
            method = getattr(solr, send_method)
        except AttributeError:
            raise AttributeError("Couldn't find the send_method. Specify either stream_file or local_index")

        self.logger.info("Indexing {} into {} using {}".format(self._queue_name,
                                                               collection,
                                                               send_method))
        if threads > 1:
            if hasattr(collection, '__call__'):
                self.logger.debug("Overwriting send_method to index_json")
                method = getattr(solr, 'index_json')
                method = partial(self._wrap_dynamic, method, collection)
            else:
                method = partial(self._wrap, method, collection)
            with ThreadPool(threads) as p:
                p.map(method, self.get_todo_items())
        else:
            for todo_file in self.get_todo_items():
                try:
                    result = method(collection, todo_file)
                    if result:
                        self.complete(todo_file)
                except SolrError:
                    self.logger.error("Error Indexing Item: {}".format(todo_file))
                    self._unlock()
                    raise