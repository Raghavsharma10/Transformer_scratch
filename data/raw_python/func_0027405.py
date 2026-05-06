def _build_catalog(self, url, xml_content):
        '''
        Recursive function to perform the DFS and yield the leaf datasets
        :param str url: URL for the current catalog
        :param str xml_content: XML Body returned from HTTP Request
        '''
        try:
            tree = etree.XML(xml_content)
        except BaseException:
            return

        # Get a list of URLs
        references = self._compile_references(url, tree)
        # Using multiple processes, make HTTP requests for each child catalog
        jobs = [self.pool.apply_async(request_xml, args=(ref,)) for ref in references]
        responses = [j.get() for j in jobs]

        # This is essentially the graph traversal step
        for i, response in enumerate(responses):
            for ds in self._build_catalog(references[i], response):
                yield ds

        # Yield the leaves
        for ds in self._yield_leaves(url, tree):
            yield ds