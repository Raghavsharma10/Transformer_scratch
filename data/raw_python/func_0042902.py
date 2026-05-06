def query(self, sql, param=None, stream=False, row_tuples=False):
        """
        RETURN LIST OF dicts
        """
        if not self.cursor:  # ALLOW NON-TRANSACTIONAL READS
            Log.error("must perform all queries inside a transaction")
        self._execute_backlog()

        try:
            if param:
                sql = expand_template(sql, quote_param(param))
            sql = self.preamble + outdent(sql)
            self.debug and Log.note("Execute SQL:\n{{sql}}", sql=indent(sql))

            self.cursor.execute(sql)
            if row_tuples:
                if stream:
                    result = self.cursor
                else:
                    result = wrap(list(self.cursor))
            else:
                columns = [utf8_to_unicode(d[0]) for d in coalesce(self.cursor.description, [])]
                if stream:
                    result = (wrap({c: utf8_to_unicode(v) for c, v in zip(columns, row)}) for row in self.cursor)
                else:
                    result = wrap([{c: utf8_to_unicode(v) for c, v in zip(columns, row)} for row in self.cursor])

            return result
        except Exception as e:
            e = Except.wrap(e)
            if "InterfaceError" in e:
                Log.error("Did you close the db connection?", e)
            Log.error("Problem executing SQL:\n{{sql|indent}}", sql=sql, cause=e, stack_depth=1)