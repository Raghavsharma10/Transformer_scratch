def searchFeaturesInDb(
            self, startIndex=0, maxResults=None,
            referenceName=None, start=None, end=None,
            parentId=None, featureTypes=None,
            name=None, geneSymbol=None):
        """
        Perform a full features query in database.

        :param startIndex: int representing first record to return
        :param maxResults: int representing number of records to return
        :param referenceName: string representing reference name, ex 'chr1'
        :param start: int position on reference to start search
        :param end: int position on reference to end search >= start
        :param parentId: string restrict search by id of parent node.
        :param name: match features by name
        :param geneSymbol: match features by gene symbol
        :return an array of dictionaries, representing the returned data.
        """
        # TODO: Refactor out common bits of this and the above count query.
        sql, sql_args = self.featuresQuery(
            startIndex=startIndex, maxResults=maxResults,
            referenceName=referenceName, start=start, end=end,
            parentId=parentId, featureTypes=featureTypes,
            name=name, geneSymbol=geneSymbol)
        sql += sqlite_backend.limitsSql(startIndex, maxResults)
        query = self._dbconn.execute(sql, sql_args)
        return sqlite_backend.sqliteRowsToDicts(query.fetchall())