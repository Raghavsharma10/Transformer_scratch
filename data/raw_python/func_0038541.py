def Vizier_xmatch(self, viz_cat, cat_name, ra_col='_RAJ2000', dec_col='_DEJ2000', radius='', group=True):
        """
        Use astroquery to pull in and cross match a catalog with sources in self.catalog
        
        Parameters
        ----------
        viz_cat: str
            The catalog string from Vizier (e.g. 'II/246' for 2MASS PSC)
        cat_name: str
            A name for the imported catalog (e.g. '2MASS')
        radius: astropy.units.quantity.Quantity
            The matching radius
        """
        # Make sure sources have been grouped
        if self.catalog.empty:
            print('Please run group_sources() before cross matching.')
            return
            
        if self._catalog_check(cat_name):
            
            # Verify the cat_name
            viz_cat = "vizier:{}".format(viz_cat)
            
            # Prep the current catalog as an astropy.QTable
            tab = at.Table.from_pandas(self.catalog)
            
            # Crossmatch with Vizier
            print("Cross matching {} sources with {} catalog. Please be patient...".format(len(tab), viz_cat))
            data = XMatch.query(cat1=tab, cat2=viz_cat, max_distance=radius or self.xmatch_radius*q.deg, colRA1='ra', colDec1='dec', colRA2=ra_col, colDec2=dec_col)
            
            # Ingest the data
            self.ingest_data(data, cat_name, 'id', ra_col=ra_col, dec_col=dec_col)
            
            # Regroup
            if group:
                self.group_sources(self.xmatch_radius)