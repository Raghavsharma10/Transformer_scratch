def ingest_data(self, data, cat_name, id_col, ra_col='_RAJ2000', dec_col='_DEJ2000', cat_loc='', append=False, count=-1):
        """
        Ingest a data file and regroup sources
        
        Parameters
        ----------
        data: str, pandas.DataFrame, astropy.table.Table
            The path to the exported VizieR data or the data table
        cat_name: str
            The name of the added catalog
        id_col: str
            The name of the column containing the unique ids
        ra_col: str
            The name of the RA column
        dec_col: str
            The name of the DEC column
        cat_loc: str
            The location of the original catalog data
        append: bool
            Append the catalog rather than replace
        count: int
            The number of table rows to add
            (This is mainly for testing purposes)
        """
        # Check if the catalog is already ingested
        if not append and cat_name in self.catalogs:
            
            print('Catalog {} already ingested.'.format(cat_name))
            
        else:
            
            if isinstance(data, str):
                cat_loc = cat_loc or data
                data = pd.read_csv(data, sep='\t', comment='#', engine='python')[:count]
                
            elif isinstance(data, pd.core.frame.DataFrame):
                cat_loc = cat_loc or type(data)
                
            elif isinstance(data, (at.QTable, at.Table)):
                cat_loc = cat_loc or type(data)
                data = pd.DataFrame(list(data), columns=data.colnames)
                
            else:
                print("Sorry, but I cannot read that data. Try an ascii file cat_loc, astropy table, or pandas data frame.")
                return
                
            # Make sure ra and dec are decimal degrees
            if isinstance(data[ra_col][0], str):
                
                crds = coord.SkyCoord(ra=data[ra_col], dec=data[dec_col], unit=(q.hour, q.deg), frame='icrs')
                data.insert(0,'dec', crds.dec)
                data.insert(0,'ra', crds.ra)
                
            elif isinstance(data[ra_col][0], float):
                
                data.rename(columns={ra_col:'ra', dec_col:'dec'}, inplace=True)
            
            else:
                print("I can't read the RA and DEC of the input data. Please try again.")
                return
                
            # Change some names
            try:
                last = len(getattr(self, cat_name)) if append else 0
                data.insert(0,'catID', ['{}_{}'.format(cat_name,n+1) for n in range(last,last+len(data))])
                data.insert(0,'dec_corr', data['dec'])
                data.insert(0,'ra_corr', data['ra'])
                data.insert(0,'source_id', np.nan)
            
                print('Ingesting {} rows from {} catalog...'.format(len(data),cat_name))
            
                # Save the raw data as an attribute
                if append:
                    setattr(self, cat_name, getattr(self, cat_name).append(data, ignore_index=True))
                
                else:
                    setattr(self, cat_name, data)
                
                # Update the history
                self.history += "\n{}: Catalog {} ingested.".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),cat_name)
                self.catalogs.update({cat_name:{'cat_loc':cat_loc, 'id_col':id_col, 'ra_col':ra_col, 'dec_col':dec_col}})
                
            except AttributeError:
                print("No catalog named '{}'. Set 'append=False' to create it.".format(cat_name))