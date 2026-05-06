def local_index(self, collection, filename, **kwargs):
        """
        :param str collection: The name of the collection for the request
        :param str filename: String file path of the file to index.

        Will index specified file into Solr. The `file` must be local to the server, this is faster than other indexing options.
        If the files are already on the servers I suggest you use this.
        For example::

            >>> solr.local_index('SolrClient_unittest',
                                       '/local/to/server/temp_file.json')
        """
        filename = os.path.abspath(filename)
        self.logger.info("Indexing {} into Solr Collection {}".format(filename, collection))

        data = {'stream.file': filename,
                'stream.contentType': 'text/json'}
        resp, con_inf = self.transport.send_request(method='GET', endpoint='update/json', collection=collection,
                                                    params=data, **kwargs)
        if resp['responseHeader']['status'] == 0:
            return True
        else:
            return False