def _relation_exists(self, connection, relation):
        """ Returns True if relation (table or view) exists in the sqlite db. Otherwise returns False.

        Args:
            connection (apsw.Connection): connection to sqlite database who stores mpr data.
            partition (orm.Partition):

        Returns:
            boolean: True if relation exists, False otherwise.

        """
        query = 'SELECT 1 FROM sqlite_master WHERE (type=\'table\' OR type=\'view\') AND name=?;'
        cursor = connection.cursor()
        cursor.execute(query, [relation])
        result = cursor.fetchall()
        return result == [(1,)]