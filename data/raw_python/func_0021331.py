def key_from_query(self, qualifier=None):
        """
        Given a Query, create a cache key.

        There are many approaches to this; here we use the simplest, which is
        to create an md5 hash of the text of the SQL statement, combined with
        stringified versions of all the bound parameters within it.

        There's a bit of a performance hit with compiling out "query.statement"
        here; other approaches include setting up an explicit cache key with a
        particular Query, then combining that with the bound parameter values.
        """
        stmt = self.with_labels().statement
        compiled = stmt.compile()
        params = compiled.params

        values = [str(compiled)]
        for k in sorted(params):
            values.append(repr(params[k]))
        key = u" ".join(values)
        return md5(key.encode('utf8')).hexdigest()