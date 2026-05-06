def get_version_records(self):
        """
        Yield RASH version information stored in DB. Latest first.

        :rtype: [VersionRecord]

        """
        keys = ['id', 'rash_version', 'schema_version', 'updated']
        sql = """
        SELECT id, rash_version, schema_version, updated
        FROM rash_info
        ORDER BY id DESC
        """
        with self.connection() as connection:
            for row in connection.execute(sql):
                yield VersionRecord(**dict(zip(keys, row)))