def getExpressionLevelById(self, expressionId):
        """
        :param expressionId: the ExpressionLevel ID
        :return: dictionary representing an ExpressionLevel object,
            or None if no match is found.
        """
        sql = ("SELECT * FROM Expression WHERE id = ?")
        query = self._dbconn.execute(sql, (expressionId,))
        try:
            return sqlite_backend.fetchOne(query)
        except AttributeError:
            raise exceptions.ExpressionLevelNotFoundException(
                expressionId)