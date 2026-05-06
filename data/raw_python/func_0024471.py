def versions(self):
        """ Read versions from the table

        The versions are kept in cache for the next reads.
        """
        if self._versions is None:
            with self.database.cursor_autocommit() as cursor:
                query = """
                SELECT number,
                       date_start,
                       date_done,
                       log,
                       addons
                FROM {}
                """.format(self.table_name)
                cursor.execute(query)
                rows = cursor.fetchall()
                versions = []
                for row in rows:
                    row = list(row)
                    # convert 'addons' to json
                    row[4] = json.loads(row[4]) if row[4] else []
                    versions.append(
                        self.VersionRecord(*row)
                    )
                self._versions = versions
        return self._versions