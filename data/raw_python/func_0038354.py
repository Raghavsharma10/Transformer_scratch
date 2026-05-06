def query(self, SQL, params='', fmt='array', fetch='all', unpack=False, export='', \
              verbose=False, use_converters=True):
        """
        Returns data satisfying the provided **SQL** script. Only SELECT or PRAGMA statements are allowed.
        Results can be returned in a variety of formats.
        For example, to extract the ra and dec of all entries in SOURCES in astropy.Table format one can write:
            data = db.query('SELECT ra, dec FROM sources', fmt='table')

        For more general SQL statements, see the modify() method.

        Parameters
        ----------
        SQL: str
            The SQL query to execute
        params: sequence
            Mimics the native parameter substitution of sqlite3
        fmt: str
            Returns the data as a dictionary, array, astropy.table, or pandas.Dataframe
            given 'dict', 'array', 'table', or 'pandas'
        fetch: str
            String indicating whether to return **all** results or just **one**
        unpack: bool
            Returns the transpose of the data
        export: str
            The file path of the ascii file to which the data should be exported
        verbose: bool
            print the data as well
        use_converters: bool
            Apply converters to columns with custom data types

        Returns
        -------
        result: (array,dict,table)
            The result of the database query
        """
        try:
            # Restrict queries to SELECT and PRAGMA statements
            if SQL.lower().startswith('select') or SQL.lower().startswith('pragma'):
                
                # Make the query explicit so that column and table names are preserved
                # Then, get the data as a dictionary
                origSQL = SQL
                try:
                    SQL, columns = self._explicit_query(SQL, use_converters=use_converters)
                    dictionary = self.dict(SQL, params).fetchall()
                except:
                    print('WARNING: Unable to use converters')
                    dictionary = self.dict(origSQL, params).fetchall()
                    
                if any(dictionary):
                    
                    # Fetch one
                    if fetch == 'one':
                        dictionary = [dictionary.pop(0)]
                        
                    # Make an Astropy table
                    table = at.Table(dictionary)
                    
                    # Reorder the columns
                    try:
                        table = table[columns]
                    except:
                        pass
                        
                    # Make an array
                    array = np.asarray(table)
                    
                    # Unpack the results if necessary (data types are not preserved)
                    if unpack: 
                        array = np.array(zip(*array))
                        
                    # print on screen
                    if verbose:
                        pprint(table)
                        
                    # print the results to file
                    if export:
                        # If .vot or .xml, assume VOTable export with votools
                        if export.lower().endswith('.xml') or export.lower().endswith('.vot'):
                            votools.dict_tovot(dictionary, export)
                            
                        # Otherwise print as ascii
                        else:
                            ii.write(table, export, Writer=ii.FixedWidthTwoLine, fill_values=[('None', '-')])
                            
                    # Or return the results
                    else:
                        if fetch == 'one':
                            dictionary, array = dictionary[0], array if unpack else np.array(list(array[0]))
                            
                        if fmt == 'table':
                            return table
                        elif fmt == 'dict':
                            return dictionary
                        elif fmt == 'pandas':
                            return table.to_pandas()
                        else:
                            return array
                            
                else:
                    return
                    
            else:
                print(
                'Queries must begin with a SELECT or PRAGMA statement. For database modifications use self.modify() method.')
                
        except IOError:
            print('Could not execute: ' + SQL)