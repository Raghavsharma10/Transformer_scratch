def update_version_records(self):
        """
        Update rash_info table if necessary.
        """
        from .__init__ import __version__ as version
        with self.connection(commit=True) as connection:
            for vrec in self.get_version_records():
                if (vrec.rash_version == version and
                    vrec.schema_version == schema_version):
                    return  # no need to insert the new one!
            connection.execute(
                'INSERT INTO rash_info (rash_version, schema_version) '
                'VALUES (?, ?)',
                [version, schema_version])