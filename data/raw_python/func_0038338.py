def add_data(self, data, table, delimiter='|', bands='', clean_up=True, rename_columns={}, column_fill={}, verbose=False):
        """
        Adds data to the specified database table. Column names must match table fields to insert,
        however order and completeness don't matter.
        
        Parameters
        ----------
        data: str, array-like, astropy.table.Table
            The path to an ascii file, array-like object, or table. The first row or element must
            be the list of column names
        table: str
            The name of the table into which the data should be inserted
        delimiter: str
            The string to use as the delimiter when parsing the ascii file
        bands: sequence
            Sequence of band to look for in the data header when digesting columns of
            multiple photometric measurements (e.g. ['MKO_J','MKO_H','MKO_K']) into individual
            rows of data for database insertion
        clean_up: bool
            Run self.clean_up()
        rename_columns: dict
            A dictionary of the {input_col_name:desired_col_name} for table columns,
            e.g. {'e_Jmag':'J_unc', 'RAJ2000':'ra'}
        column_fill: dict
            A dictionary of the column name and value to fill, e.g. {'instrument_id':2, 'band':'2MASS.J'}
        verbose: bool
          Print diagnostic messages
        """
        # Store raw entry
        entry, del_records = data, []

        # Digest the ascii file into table
        if isinstance(data, str) and os.path.isfile(data):
            data = ii.read(data, delimiter=delimiter)

        # Or read the sequence of data elements into a table
        elif isinstance(data, (list, tuple, np.ndarray)):
            data = ii.read(['|'.join(map(str, row)) for row in data], data_start=1, delimiter='|')
            
        # Or convert pandas dataframe to astropy table
        elif isinstance(data, pd.core.frame.DataFrame):
            data = at.Table.from_pandas(data)
            
        # Or if it's already an astropy table
        elif isinstance(data, at.Table):
            pass
            
        else:
            data = None
            
        if data:
            
            # Rename columns
            if isinstance(rename_columns,str):
                rename_columns = astrocat.default_rename_columns(rename_columns)
            for input_name,new_name in rename_columns.items():
                data.rename_column(input_name,new_name)
                
            # Add column fills
            if isinstance(column_fill,str):
                column_fill = astrocat.default_column_fill(column_fill)
            for colname,fill_value in column_fill.items():
                data[colname] = [fill_value]*len(data)
                
            # Get list of all columns and make an empty table for new records
            metadata = self.query("PRAGMA table_info({})".format(table), fmt='table')
            columns, types, required = [np.array(metadata[n]) for n in ['name', 'type', 'notnull']]
            new_records = at.Table(names=columns, dtype=[type_dict[t] for t in types])
            
            # Fix column dtypes and blanks
            for col in data.colnames:
                
                # Convert data dtypes to those of the existing table
                try:
                    temp = data[col].astype(new_records[col].dtype)
                    data.replace_column(col, temp)
                except KeyError:
                    continue
                    
            # If a row contains photometry for multiple bands, use the *multiband argument and execute this
            if bands and table.lower() == 'photometry':
                
                # Pull out columns that are band names
                for b in list(set(bands) & set(data.colnames)):
                    
                    try:
                        # Get the repeated data plus the band data and rename the columns
                        band = data[list(set(columns) & set(data.colnames)) + [b, b + '_unc']]
                        for suf in ['', '_unc']:
                            band.rename_column(b+suf, 'magnitude'+suf)
                            temp = band['magnitude'+suf].astype(new_records['magnitude'+suf].dtype)
                            band.replace_column('magnitude'+suf, temp)
                        band.add_column(at.Column([b] * len(band), name='band', dtype='O'))
                        
                        # Add the band data to the list of new_records
                        new_records = at.vstack([new_records, band])
                        
                    except IOError:
                        pass
                        
            else:
                # Inject data into full database table format
                new_records = at.vstack([new_records, data])[new_records.colnames]
                
            # Reject rows that fail column requirements, e.g. NOT NULL fields like 'source_id'
            for r in columns[np.where(np.logical_and(required, columns != 'id'))]:
                # Null values...
                new_records = new_records[np.where(new_records[r])]
                
                # Masked values...
                try:
                    new_records = new_records[~new_records[r].mask]
                except:
                    pass
                
                # NaN values...
                if new_records.dtype[r] in (int, float):
                    new_records = new_records[~np.isnan(new_records[r])]
                    
            # For spectra, try to populate the table by reading the FITS header
            if table.lower() == 'spectra':
                for n, new_rec in enumerate(new_records):
                    
                    # Convert relative path to absolute path
                    relpath = new_rec['spectrum']
                    if relpath.startswith('$'):
                        abspath = os.popen('echo {}'.format(relpath.split('/')[0])).read()[:-1]
                        if abspath:
                            new_rec['spectrum'] = relpath.replace(relpath.split('/')[0], abspath)
                            
                    # Test if the file exists and try to pull metadata from the FITS header
                    if os.path.isfile(new_rec['spectrum']):
                        new_records[n]['spectrum'] = relpath
                        new_records[n] = _autofill_spec_record(new_rec)
                    else:
                        print('Error adding the spectrum at {}'.format(new_rec['spectrum']))
                        del_records.append(n)
                        
                # Remove bad records from the table
                new_records.remove_rows(del_records)
                
            # For images, try to populate the table by reading the FITS header
            if table.lower() == 'images':
                for n, new_rec in enumerate(new_records):
                    
                    # Convert relative path to absolute path
                    relpath = new_rec['image']
                    if relpath.startswith('$'):
                        abspath = os.popen('echo {}'.format(relpath.split('/')[0])).read()[:-1]
                        if abspath:
                            new_rec['image'] = relpath.replace(relpath.split('/')[0], abspath)

                    # Test if the file exists and try to pull metadata from the FITS header
                    if os.path.isfile(new_rec['image']):
                        new_records[n]['image'] = relpath
                        new_records[n] = _autofill_spec_record(new_rec)
                    else:
                        print('Error adding the image at {}'.format(new_rec['image']))
                        del_records.append(n)
                        
                # Remove bad records from the table
                new_records.remove_rows(del_records)
                
            # Get some new row ids for the good records
            rowids = self._lowest_rowids(table, len(new_records))
            
            # Add the new records
            keepers, rejects = [], []
            for N, new_rec in enumerate(new_records):
                new_rec = list(new_rec)
                new_rec[0] = rowids[N]
                for n, col in enumerate(new_rec):
                    if type(col) == np.int64 and sys.version_info[0] >= 3:
                        # Fix for Py3 and sqlite3 issue with numpy types
                        new_rec[n] = col.item()
                    if type(col) == np.ma.core.MaskedConstant:
                        new_rec[n] = None
                        
                try:
                    self.modify("INSERT INTO {} VALUES({})".format(table, ','.join('?'*len(columns))), new_rec, verbose=verbose)
                    keepers.append(N)
                except IOError:
                    rejects.append(N)
                    
                new_records[N]['id'] = rowids[N]
                
            # Make tables of keepers and rejects
            rejected = new_records[rejects]
            new_records = new_records[keepers]
            
            # Print a table of the new records or bad news
            if new_records:
                print("\033[1;32m{} new records added to the {} table.\033[1;m".format(len(new_records), table.upper()))
                new_records.pprint()
                
            if rejected:
                print("\033[1;31m{} records rejected from the {} table.\033[1;m".format(len(rejected), table.upper()))
                rejected.pprint()
                
            # Run table clean up
            if clean_up:
                self.clean_up(table, verbose)

        else:
            print('Please check your input: {}'.format(entry))