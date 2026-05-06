def scan_index(self, query: Union[Dict[str, str], None] = None) -> List[Dict[str, str]]:
        """Scan the index with the query.

        Will return any number of results above 10'000. Important to note is, that
        all the data is loaded into memory at once and returned. This works only with small
        data sets. Use scroll otherwise which returns a generator to cycle through the resources
        in set chunks.

        :param query: The query used to scan the index. Default None will return the entire index.
        :returns list of dicts: The list of dictionaries contains all the documents without metadata.
        """
        if query is None:
            query = self.match_all
        logging.info('Download all documents from index %s with query %s.', self.index, query)
        results = list()
        data = scan(self.instance, index=self.index, doc_type=self.doc_type, query=query)
        for items in data:
            if '_source' in items:
                results.append(items['_source'])
            else:
                results.append(items)
        return results