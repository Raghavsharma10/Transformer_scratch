def inventory(self, source_id, fetch=False, fmt='table'):
        """
        Prints a summary of all objects in the database. Input string or list of strings in **ID** or **unum**
        for specific objects.

        Parameters
        ----------
        source_id: int
            The id from the SOURCES table whose data across all tables is to be printed.
        fetch: bool
            Return the results.
        fmt: str
            Returns the data as a dictionary, array, or astropy.table given 'dict', 'array', or 'table'

        Returns
        -------
        data_tables: dict
            Returns a dictionary of astropy tables with the table name as the keys.

        """
        data_tables = {}

        t = self.query("SELECT * FROM sqlite_master WHERE type='table'", fmt='table')
        all_tables = t['name'].tolist()
        for table in ['sources'] + [t for t in all_tables if
                                    t not in ['sources', 'sqlite_sequence']]:

            try:

                # Get the columns, pull out redundant ones, and query the table for this source's data
                t = self.query("PRAGMA table_info({})".format(table), fmt='table')
                columns = np.array(t['name'])
                types = np.array(t['type'])

                if table == 'sources' or 'source_id' in columns:

                    # If printing, only get simple data types and exclude redundant 'source_id' for nicer printing
                    if not fetch:
                        columns = columns[
                            ((types == 'REAL') | (types == 'INTEGER') | (types == 'TEXT')) & (columns != 'source_id')]

                    # Query the table
                    try:
                        id = 'id' if table.lower() == 'sources' else 'source_id'
                        data = self.query(
                            "SELECT {} FROM {} WHERE {}={}".format(','.join(columns), table, id, source_id),
                            fmt='table')

                        if not data and table.lower() == 'sources':
                            print(
                            'No source with id {}. Try db.search() to search the database for a source_id.'.format(
                                source_id))

                    except:
                        data = None

                    # If there's data for this table, save it
                    if data:
                        if fetch:
                            data_tables[table] = self.query(
                                "SELECT {} FROM {} WHERE {}={}".format(','.join(columns), table, id, source_id), \
                                fetch=True, fmt=fmt)
                        else:
                            data = data[[c.lower() for c in columns]]
                            pprint(data, title=table.upper())

                else:
                    pass

            except:
                print('Could not retrieve data from {} table.'.format(table.upper()))

        if fetch: return data_tables