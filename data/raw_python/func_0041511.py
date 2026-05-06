def exists(c_table_cd: str, tables: I2B2Tables) -> int:
        """ Return the number of records that exist with the table code.
        - Ideally this should be zero or one, but the default table doesn't have a key

        :param c_table_cd: key to test
        :param tables:
        :return: number of records found
        """
        conn = tables.ont_connection
        table = tables.schemes
        return bool(list(conn.execute(table.select().where(table.c.c_table_cd == c_table_cd))))