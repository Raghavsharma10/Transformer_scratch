def searchRnaQuantificationsInDb(
            self, rnaQuantificationId=""):
        """
        :param rnaQuantificationId: string restrict search by id
        :return an array of dictionaries, representing the returned data.
        """
        sql = ("SELECT * FROM RnaQuantification")
        sql_args = ()
        if len(rnaQuantificationId) > 0:
            sql += " WHERE id = ? "
            sql_args += (rnaQuantificationId,)
        query = self._dbconn.execute(sql, sql_args)
        try:
            return sqlite_backend.iterativeFetch(query)
        except AttributeError:
            raise exceptions.RnaQuantificationNotFoundException(
                rnaQuantificationId)