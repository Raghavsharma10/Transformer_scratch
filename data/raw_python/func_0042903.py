def column_query(self, sql, param=None):
        """
        RETURN RESULTS IN [column][row_num] GRID
        """
        self._execute_backlog()
        try:
            old_cursor = self.cursor
            if not old_cursor:  # ALLOW NON-TRANSACTIONAL READS
                self.cursor = self.db.cursor()
                self.cursor.execute("SET TIME_ZONE='+00:00'")
                self.cursor.close()
                self.cursor = self.db.cursor()

            if param:
                sql = expand_template(sql, quote_param(param))
            sql = self.preamble + outdent(sql)
            self.debug and Log.note("Execute SQL:\n{{sql}}", sql=indent(sql))

            self.cursor.execute(sql)
            grid = [[utf8_to_unicode(c) for c in row] for row in self.cursor]
            # columns = [utf8_to_unicode(d[0]) for d in coalesce(self.cursor.description, [])]
            result = transpose(*grid)

            if not old_cursor:  # CLEANUP AFTER NON-TRANSACTIONAL READS
                self.cursor.close()
                self.cursor = None

            return result
        except Exception as e:
            if isinstance(e, InterfaceError) or e.message.find("InterfaceError") >= 0:
                Log.error("Did you close the db connection?", e)
            Log.error("Problem executing SQL:\n{{sql|indent}}", sql=sql, cause=e, stack_depth=1)