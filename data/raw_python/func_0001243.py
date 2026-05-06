def schemas(self):
        """
        Get a listing of all non-system schemas (prefixed with 'pg_') that
        exist in the database.
        """
        sql = """SELECT schema_name FROM information_schema.schemata
                 ORDER BY schema_name"""
        schemas = self.query(sql).fetchall()
        return [s[0] for s in schemas if s[0][:3] != "pg_"]