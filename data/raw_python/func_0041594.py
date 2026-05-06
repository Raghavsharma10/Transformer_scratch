def _delete_upload_id(conn: Connection, table: Table, upload_id: int) -> int:
        """Remove all table records with the supplied upload_id

        :param conn: sql connection
        :param table: table to modify
        :param upload_id: target upload_id
        :return: number of records removed
        """
        return conn.execute(delete(table).where(table.c.upload_id == upload_id)).rowcount if upload_id else 0