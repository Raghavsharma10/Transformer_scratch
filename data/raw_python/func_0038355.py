def references(self, criteria, publications='publications', column_name='publication_shortname', fetch=False):
        """
        Do a reverse lookup on the **publications** table. Will return every entry that matches that reference.

        Parameters
        ----------
        criteria: int or str
            The id from the PUBLICATIONS table whose data across all tables is to be printed.
        publications: str
            Name of the publications table
        column_name: str
            Name of the reference column in other tables
        fetch: bool
            Return the results.

        Returns
        -------
        data_tables: dict
             Returns a dictionary of astropy tables with the table name as the keys.

        """

        data_tables = dict()

        # If an ID is provided but the column name is publication shortname, grab the shortname
        if isinstance(criteria, type(1)) and column_name == 'publication_shortname':
            t = self.query("SELECT * FROM {} WHERE id={}".format(publications, criteria), fmt='table')
            if len(t) > 0:
                criteria = t['shortname'][0]
            else:
                print('No match found for {}'.format(criteria))
                return

        t = self.query("SELECT * FROM sqlite_master WHERE type='table'", fmt='table')
        all_tables = t['name'].tolist()
        for table in ['sources'] + [t for t in all_tables if
                                    t not in ['publications', 'sqlite_sequence', 'sources']]:

            # Get the columns, pull out redundant ones, and query the table for this source's data
            t = self.query("PRAGMA table_info({})".format(table), fmt='table')
            columns = np.array(t['name'])
            types = np.array(t['type'])

            # Only get simple data types and exclude redundant ones for nicer printing
            columns = columns[
                ((types == 'REAL') | (types == 'INTEGER') | (types == 'TEXT')) & (columns != column_name)]

            # Query the table
            try:
                data = self.query("SELECT {} FROM {} WHERE {}='{}'".format(','.join(columns), table,
                                                                         column_name, criteria), fmt='table')
            except:
                data = None

            # If there's data for this table, save it
            if data:
                if fetch:
                    data_tables[table] = self.query(
                        "SELECT {} FROM {} WHERE {}='{}'".format(
                            ','.join(columns), table, column_name, criteria), fmt='table', fetch=True)
                else:
                    data = data[[c.lower() for c in columns]]  # force lowercase since astropy.Tables have all lowercase
                    pprint(data, title=table.upper())

        if fetch: return data_tables