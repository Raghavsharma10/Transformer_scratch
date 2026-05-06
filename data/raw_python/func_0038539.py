def SDSS_spectra_query(self, cat_name, ra, dec, radius, group=True, **kwargs):
        """
        Use astroquery to search SDSS for sources within a search cone
        
        Parameters
        ----------
        cat_name: str
            A name for the imported catalog (e.g. '2MASS')
        ra: astropy.units.quantity.Quantity
            The RA of the center of the cone search
        dec: astropy.units.quantity.Quantity
            The Dec of the center of the cone search
        radius: astropy.units.quantity.Quantity
            The radius of the cone search
        """
        # Verify the cat_name
        if self._catalog_check(cat_name):
            
            # Prep the current catalog as an astropy.QTable
            tab = at.Table.from_pandas(self.catalog)
            
            # Cone search Vizier
            print("Searching SDSS for sources within {} of ({}, {}). Please be patient...".format(viz_cat, radius, ra, dec))
            crds = coord.SkyCoord(ra=ra, dec=dec, frame='icrs')
            try:
                data = SDSS.query_region(crds, spectro=True, radius=radius)
            except:
                print("No data found in SDSS within {} of ({}, {}).".format(viz_cat, radius, ra, dec))
                return
            
            # Ingest the data
            self.ingest_data(data, cat_name, 'id', ra_col=ra_col, dec_col=dec_col)
            
            # Regroup
            if len(self.catalogs)>1 and group:
                self.group_sources(self.xmatch_radius)