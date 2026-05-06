def search(self, criterion, table, columns='', fetch=False, radius=1/60., use_converters=False, sql_search=False):
        """
        General search method for tables. For (ra,dec) input in decimal degrees,
        i.e. (12.3456,-65.4321), returns all sources within 1 arcminute, or the specified radius.
        For string input, i.e. 'vb10', returns all sources with case-insensitive partial text
        matches in columns with 'TEXT' data type. For integer input, i.e. 123, returns all
        exact matches of columns with INTEGER data type.

        Parameters
        ----------
        criterion: (str, int, sequence, tuple)
            The text, integer, coordinate tuple, or sequence thereof to search the table with.
        table: str
            The name of the table to search
        columns: sequence
            Specific column names to search, otherwise searches all columns
        fetch: bool
            Return the results of the query as an Astropy table
        radius: float
            Radius in degrees in which to search for objects if using (ra,dec). Default: 1/60 degree
        use_converters: bool
            Apply converters to columns with custom data types
        sql_search: bool
            Perform the search by coordinates in a box defined within the SQL commands, rather than with true angular
            separations. Faster, but not a true radial search.
        """

        # Get list of columns to search and format properly
        t = self.query("PRAGMA table_info({})".format(table), unpack=True, fmt='table')
        all_columns = t['name'].tolist()
        types = t['type'].tolist()
        columns = columns or all_columns
        columns = np.asarray([columns] if isinstance(columns, str) else columns)

        # Separate good and bad columns and corresponding types
        badcols = columns[~np.in1d(columns, all_columns)]
        columns = columns[np.in1d(columns, all_columns)]
        columns = np.array([c for c in all_columns if c in columns])
        types = np.array([t for c, t in zip(all_columns, types) if c in columns])[np.in1d(columns, all_columns)]
        for col in badcols:
            print("'{}' is not a column in the {} table.".format(col, table.upper()))

        # Coordinate search
        if sys.version_info[0] == 2:
            str_check = (str, unicode)
        else:
            str_check = str

        results = ''

        if isinstance(criterion, (tuple, list, np.ndarray)):
            try:
                if sql_search:
                    q = "SELECT * FROM {} WHERE ra BETWEEN ".format(table) \
                        + str(criterion[0] - radius) + " AND " \
                        + str(criterion[0] + radius) + " AND dec BETWEEN " \
                        + str(criterion[1] - radius) + " AND " \
                        + str(criterion[1] + radius)
                    results = self.query(q, fmt='table')
                else:
                    t = self.query('SELECT id,ra,dec FROM sources', fmt='table')
                    df = t.to_pandas()
                    df[['ra', 'dec']] = df[['ra', 'dec']].apply(pd.to_numeric)  # convert everything to floats
                    mask = df['ra'].isnull()
                    df = df[~mask]

                    df['theta'] = df.apply(ang_sep, axis=1, args=(criterion[0], criterion[1]))
                    good = df['theta'] <= radius

                    if sum(good) > 0:
                        params = ", ".join(['{}'.format(s) for s in df[good]['id'].tolist()])
                        try:
                            results = self.query('SELECT * FROM {} WHERE source_id IN ({})'.format(table, params),
                                                 fmt='table')
                        except:
                            results = self.query('SELECT * FROM {} WHERE id IN ({})'.format(table, params),
                                                 fmt='table')
            except:
                print("Could not search {} table by coordinates {}. Try again.".format(table.upper(), criterion))

        # Text string search of columns with 'TEXT' data type
        elif isinstance(criterion, str_check) and any(columns) and 'TEXT' in types:
            try:
                q = "SELECT * FROM {} WHERE {}".format(table, ' OR '.join([r"REPLACE(" + c + r",' ','') like '%" \
                     + criterion.replace(' ', '') + r"%'" for c, t in zip(columns,types[np.in1d(columns, all_columns)]) \
                     if t == 'TEXT']))
                results = self.query(q, fmt='table', use_converters=use_converters)
            except:
                print("Could not search {} table by string {}. Try again.".format(table.upper(), criterion))

        # Integer search of columns with 'INTEGER' data type
        elif isinstance(criterion, int):
            try:
                q = "SELECT * FROM {} WHERE {}".format(table, ' OR '.join(['{}={}'.format(c, criterion) \
                     for c, t in zip(columns, types[np.in1d(columns, all_columns)]) if t == 'INTEGER']))
                results = self.query(q, fmt='table', use_converters=use_converters)
            except:
                print("Could not search {} table by id {}. Try again.".format(table.upper(), criterion))

        # Problem!
        else:
            print("Could not search {} table by '{}'. Try again.".format(table.upper(), criterion))

        # Print or return the results
        if fetch:
            return results or at.Table(names=columns, dtype=[type_dict[t] for t in types], masked=True)
        else:
            if results: 
                pprint(results, title=table.upper())
            else:
                print("No results found for {} in the {} table.".format(criterion, table.upper()))