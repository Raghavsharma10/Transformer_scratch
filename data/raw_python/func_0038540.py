def Vizier_query(self, viz_cat, cat_name, ra, dec, radius, ra_col='RAJ2000', dec_col='DEJ2000', columns=["**"], append=False, group=True, **kwargs):
        """
        Use astroquery to search a catalog for sources within a search cone
        
        Parameters
        ----------
        viz_cat: str
            The catalog string from Vizier (e.g. 'II/246' for 2MASS PSC)
        cat_name: str
            A name for the imported catalog (e.g. '2MASS')
        ra: astropy.units.quantity.Quantity
            The RA of the center of the cone search
        dec: astropy.units.quantity.Quantity
            The Dec of the center of the cone search
        radius: astropy.units.quantity.Quantity
            The radius of the cone search
        ra_col: str
            The name of the RA column in the raw catalog
        dec_col: str
            The name of the Dec column in the raw catalog
        columns: sequence
            The list of columns to pass to astroquery
        append: bool
            Append the catalog rather than replace
        """
        # Verify the cat_name
        if self._catalog_check(cat_name, append=append):
            
            # Cone search Vizier
            print("Searching {} for sources within {} of ({}, {}). Please be patient...".format(viz_cat, radius, ra, dec))
            crds = coord.SkyCoord(ra=ra, dec=dec, frame='icrs')
            V = Vizier(columns=columns, **kwargs)
            V.ROW_LIMIT = -1
            
            try:
                data = V.query_region(crds, radius=radius, catalog=viz_cat)[0]
            except:
                print("No data found in {} within {} of ({}, {}).".format(viz_cat, radius, ra, dec))
                return
            
            # Ingest the data
            self.ingest_data(data, cat_name, 'id', ra_col=ra_col, dec_col=dec_col, cat_loc=viz_cat, append=append)
            
            # Regroup
            if len(self.catalogs)>1 and group:
                self.group_sources(self.xmatch_radius)