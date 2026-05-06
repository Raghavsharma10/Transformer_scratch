def correct_offsets(self, cat_name, truth='ACS'):
        """
        Function to determine systematic, linear offsets between catalogs
        
        FUTURE -- do this with TweakReg, which also accounts for rotation/scaling
        See thread at https://github.com/spacetelescope/drizzlepac/issues/77
        
        Parameters
        ----------
        cat_name: str
            Name of catalog to correct
        truth: str
            The catalog to measure against
        """
        # Must be grouped!
        if not self.xmatch_radius:
            
            print("Please run group_sources() before running correct_offsets().")
            
        else:
            
            # First, remove any previous catalog correction
            self.catalog.loc[self.catalog['cat_name']==cat_name, 'ra_corr'] = self.catalog.loc[self.catalog['cat_name']==cat_name, '_RAJ2000']
            self.catalog.loc[self.catalog['cat_name']==cat_name, 'dec_corr'] = self.catalog.loc[self.catalog['cat_name']==cat_name, '_DEJ2000']
            
            # Copy the catalog
            onc_gr = self.catalog.copy()
            
            # restrict to one-to-one matches, sort by oncID so that matches are paired
            o2o_new = onc_gr.loc[(onc_gr['oncflag'].str.contains('o')) & (onc_gr['cat_name'] == cat_name) ,:].sort_values('oncID')
            o2o_old = onc_gr.loc[(onc_gr['oncID'].isin(o2o_new['oncID']) & (onc_gr['cat_name'] == truth)), :].sort_values('oncID')
            
            # get coords
            c_o2o_new = SkyCoord(o2o_new.loc[o2o_new['cat_name'] == cat_name, 'ra_corr'],\
                                 o2o_new.loc[o2o_new['cat_name'] == cat_name, 'dec_corr'], unit='degree')
            c_o2o_old = SkyCoord(o2o_old.loc[o2o_old['cat_name'] == truth, 'ra_corr'],\
                                 o2o_old.loc[o2o_old['cat_name'] == truth, 'dec_corr'], unit='degree')
                             
            print(len(c_o2o_old), 'one-to-one matches found!')
            
            if len(c_o2o_old)>0:
                
                delta_ra = []
                delta_dec = []
                
                for i in range(len(c_o2o_old)):
                    # offsets FROM ACS TO new catalog
                    ri, di = c_o2o_old[i].spherical_offsets_to(c_o2o_new[i])
                    
                    delta_ra.append(ri.arcsecond)
                    delta_dec.append(di.arcsecond)
                    
                    progress_meter((i+1)*100./len(c_o2o_old))
                    
                delta_ra = np.array(delta_ra)
                delta_dec = np.array(delta_dec)
                
                print('\n')
                
                # fit a gaussian
                mu_ra, std_ra = norm.fit(delta_ra)
                mu_dec, std_dec = norm.fit(delta_dec)
                
                # Fix precision
                mu_ra = round(mu_ra, 6)
                mu_dec = round(mu_dec, 6)
                
                # Update the coordinates of the appropriate sources
                print('Shifting {} sources by {}" in RA and {}" in Dec...'.format(cat_name,mu_ra,mu_dec))
                self.catalog.loc[self.catalog['cat_name']==cat_name, 'ra_corr'] += mu_ra
                self.catalog.loc[self.catalog['cat_name']==cat_name, 'dec_corr'] += mu_dec
                
                # Update history
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.history += "\n{}: {} sources shifted by {} deg in RA and {} deg in Declination.".format(now, cat_name, mu_ra, mu_dec)
                
                # Regroup the sources since many have moved
                self.group_sources(self.xmatch_radius)
                
            else:
                
                print('Cannot correct offsets in {} sources.'.format(cat_name))