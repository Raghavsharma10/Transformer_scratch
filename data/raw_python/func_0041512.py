def del_records(c_table_cd: str, tables: I2B2Tables) -> int:
        """ Delete all records with c_table_code

        :param c_table_cd: key to delete
        :param tables:
        :return: number of records deleted
        """
        conn = tables.ont_connection
        table = tables.schemes
        return conn.execute(table.delete().where(table.c.c_table_cd == c_table_cd)).rowcount