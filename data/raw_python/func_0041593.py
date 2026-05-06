def _delete_sourcesystem_cd(conn: Connection, table: Table, sourcesystem_cd: str) -> int:
        """ Remove all table records with the supplied upload_id

        :param conn: sql connection
        :param table: table to modify
        :param sourcesystem_cd: target sourcesystem code
        :return: number of records removed
        """
        return conn.execute(delete(table).where(table.c.sourcesystem_cd == sourcesystem_cd)).rowcount \
            if sourcesystem_cd else 0