def getFeatureById(self, featureId):
        """
        Fetch feature by featureID.

        :param featureId: the FeatureID as found in GFF3 records
        :return: dictionary representing a feature object,
            or None if no match is found.
        """
        sql = "SELECT * FROM FEATURE WHERE id = ?"
        query = self._dbconn.execute(sql, (featureId,))
        ret = query.fetchone()
        if ret is None:
            return None
        return sqlite_backend.sqliteRowToDict(ret)