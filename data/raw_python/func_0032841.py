def version(self):
        """What's the version of this database? Found in metadata attached
        by datacache when creating this database."""
        query = "SELECT version FROM %s" % METADATA_TABLE_NAME
        cursor = self.connection.execute(query)
        version = cursor.fetchone()
        if not version:
            return 0
        else:
            return int(version[0])