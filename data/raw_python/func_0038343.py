def _explicit_query(self, SQL, use_converters=True):
        """
        Sorts the column names so they are returned in the same order they are queried. Also turns
        ambiguous SELECT statements into explicit SQLite language in case column names are not unique.

        HERE BE DRAGONS!!! Bad bad bad. This method needs to be reworked.

        Parameters
        ----------
        SQL: str
            The SQLite query to parse
        use_converters: bool
            Apply converters to columns with custom data types

        Returns
        -------
        (SQL, columns): (str, sequence)
            The new SQLite string to use in the query and the ordered column names

        """
        try:
            # If field names are given, sort so that they come out in the same order they are fetched
            if 'select' in SQL.lower() and 'from' in SQL.lower():

                # Make a dictionary of the table aliases
                tdict = {}
                from_clause = SQL.lower().split('from ')[-1].split(' where')[0]
                tables = [j for k in [i.split(' on ') for i in from_clause.split(' join ')] for j in k if '=' not in j]

                for t in tables:
                    t = t.replace('as', '')
                    try:
                        name, alias = t.split()
                        tdict[alias] = name
                    except:
                        tdict[t] = t

                # Get all the column names and dtype placeholders
                columns = \
                SQL.replace(' ', '').lower().split('distinct' if 'distinct' in SQL.lower() else 'select')[1].split(
                    'from')[0].split(',')

                # Replace * with the field names
                for n, col in enumerate(columns):
                    if '.' in col:
                        t, col = col.split('.')
                    else:
                        t = tables[0]

                    if '*' in col:
                        col = np.array(self.list("PRAGMA table_info({})".format(tdict.get(t))).fetchall()).T[1]
                    else:
                        col = [col]

                    columns[n] = ["{}.{}".format(t, c) if len(tables) > 1 else c for c in col]

                # Flatten the list of columns and dtypes
                columns = [j for k in columns for j in k]

                # Get the dtypes
                dSQL = "SELECT " \
                       + ','.join(["typeof({})".format(col) for col in columns]) \
                       + ' FROM ' + SQL.replace('from', 'FROM').split('FROM')[-1]
                if use_converters:
                    dtypes = [None] * len(columns)
                else:
                    dtypes = self.list(dSQL).fetchone()

                # Reconstruct SQL query
                SQL = "SELECT {}".format('DISTINCT ' if 'distinct' in SQL.lower() else '') \
                      + (','.join(["{0} AS '{0}'".format(col) for col in columns]) \
                             if use_converters else ','.join(["{1}{0}{2} AS '{0}'".format(col,
                                                                                          'CAST(' if dt != 'null' else '',
                                                                                          ' AS {})'.format(
                                                                                              dt) if dt != 'null' else '') \
                                                              for dt, col in zip(dtypes, columns)])) \
                      + ' FROM ' \
                      + SQL.replace('from', 'FROM').split('FROM')[-1]

            elif 'pragma' in SQL.lower():
                columns = ['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']

            return SQL, columns

        except:
            return SQL, ''