def execute(self):
        """ Executes all sql statements from bundle.sql. """
        from ambry.mprlib import execute_sql

        execute_sql(self._bundle.library, self.record_content)