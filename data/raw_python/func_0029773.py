def search(self, search_phrase, limit=None):
        """ Finds identifiers by search phrase.

        Args:
            search_phrase (str or unicode):
            limit (int, optional): how many results to return. None means without limit.

        Returns:
            list of IdentifierSearchResult instances.

        """

        query_parts = [
            'SELECT identifier, type, name, similarity(name, :word) AS sml',
            'FROM identifier_index',
            'WHERE name % :word',
            'ORDER BY sml DESC, name']

        query_params = {
            'word': search_phrase}

        if limit:
            query_parts.append('LIMIT :limit')
            query_params['limit'] = limit

        query_parts.append(';')

        query = text('\n'.join(query_parts))

        self.backend.library.database.set_connection_search_path()

        results = self.execute(query, **query_params).fetchall()

        for result in results:
            vid, type, name, score = result
            yield IdentifierSearchResult(
                score=score, vid=vid,
                type=type, name=name)