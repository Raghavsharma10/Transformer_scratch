def searchExpressionLevelsInDb(
            self, rnaQuantId, names=[], threshold=0.0, startIndex=0,
            maxResults=0):
        """
        :param rnaQuantId: string restrict search by quantification id
        :param threshold: float minimum expression values to return
        :return an array of dictionaries, representing the returned data.
        """
        sql = ("SELECT * FROM Expression WHERE "
               "rna_quantification_id = ? "
               "AND expression > ? ")
        sql_args = (rnaQuantId, threshold)
        if len(names) > 0:
            sql += "AND name in ("
            sql += ",".join(['?' for name in names])
            sql += ") "
            for name in names:
                sql_args += (name,)
        sql += sqlite_backend.limitsSql(
            startIndex=startIndex, maxResults=maxResults)
        query = self._dbconn.execute(sql, sql_args)
        return sqlite_backend.iterativeFetch(query)