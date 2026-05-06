def add_source(self, ra, dec, flag='', radius=10*q.arcsec):
        """
        Add a source to the catalog manually and find data in existing catalogs
        
        Parameters
        ----------
        ra: astropy.units.quantity.Quantity
            The RA of the source
        dec: astropy.units.quantity.Quantity
            The Dec of the source
        flag: str
            A flag for the source
        radius: float
            The cross match radius for the list of catalogs
        """
        # Get the id
        id = int(len(self.catalog)+1)
        
        # Check the coordinates
        ra = ra.to(q.deg)
        dec = dec.to(q.deg)
        datasets = 0
        
        # Search the catalogs for this source
        for cat_name,params in self.catalogs.items():
            self.Vizier_query(params['cat_loc'], cat_name, ra, dec, radius, ra_col=params['ra_col'], dec_col=params['dec_col'], append=True, group=False)
            
        # Add the source to the catalog
        self.catalog = self.catalog.append([id, ra.value, dec.value, flag, datasets], ignore_index=True)