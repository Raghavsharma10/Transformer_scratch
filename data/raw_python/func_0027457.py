def execute(self, fetchall=False, fetchone=False, use_labels=True):
        """
        :param fetchall: get all rows
        :param fetchone:  get only one row
        :param use_labels: prefix row columns names by the table name
        :return:
        """
        query = self.get_query(use_labels=use_labels)
        if fetchall:
            return flask.g.db_conn.execute(query).fetchall()
        elif fetchone:
            return flask.g.db_conn.execute(query).fetchone()