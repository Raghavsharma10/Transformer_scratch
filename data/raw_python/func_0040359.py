def delete_upload_id(cls, tables: I2B2Tables, upload_id: int) -> int:
        """
        Delete all observation_fact records with the supplied upload_id
        :param tables: i2b2 sql connection
        :param upload_id: upload identifier to remove
        :return: number or records that were deleted
        """
        return cls._delete_upload_id(tables.crc_connection, tables.observation_fact, upload_id)