def group_sources(self, radius='', plot=False):
        """
        Calculate the centers of the point clusters given the
        radius and minimum number of points

        Parameters
        ----------
        coords: array-like
            The list of (x,y) coordinates of all clicks
        radius: int
            The distance threshold in degrees for cluster membership
            [default of 0.36 arcseconds]

        Returns
        -------
        np.ndarray
            An array of the cluster centers
        """
        if len(self.catalogs)==0:
            print("No catalogs to start grouping! Add one with the ingest_data() method first.")
            
        else:
            
            # Gather the catalogs
            print('Grouping sources from the following catalogs:',list(self.catalogs.keys()))
            cats = pd.concat([getattr(self, cat_name) for cat_name in self.catalogs])
            
            # Clear the source grouping
            cats['oncID'] = np.nan
            cats['oncflag'] = ''
            self.xmatch_radius = radius if isinstance(radius,(float,int)) else self.xmatch_radius
            
            # Make a list of the coordinates of each catalog row
            coords = cats[['ra_corr','dec_corr']].values
            
            # Perform DBSCAN to find clusters
            db = DBSCAN(eps=self.xmatch_radius, min_samples=1, n_jobs=-1).fit(coords)
            
            # Group the sources
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            source_ids = db.labels_+1
            unique_source_ids = list(set(source_ids))
            self.n_sources = len(unique_source_ids)
            
            # Get the average coordinates of all clusters
            unique_coords = np.asarray([np.mean(coords[source_ids==id], axis=0) for id in list(set(source_ids))])
            
            # Generate a source catalog
            self.catalog = pd.DataFrame(columns=('id','ra','dec','flag','datasets'))
            self.catalog['id'] = unique_source_ids
            self.catalog[['ra','dec']] = unique_coords
            self.catalog['flag'] = [None]*len(unique_source_ids)
            # self.catalog['flag'] = ['d{}'.format(i) if i>1 else '' for i in Counter(source_ids).values()]
            self.catalog['datasets'] = Counter(source_ids).values()
            
            # Update history
            self.history += "\n{}: Catalog grouped with radius {} arcsec.".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.xmatch_radius)
            
            # Update the source_ids in each catalog
            cats['source_id'] = source_ids
            for cat_name in self.catalogs:
                
                # Get the source_ids for the catalog
                cat_source_ids = cats.loc[cats['catID'].str.startswith(cat_name)]['source_id']
                
                # Get the catalog
                cat = getattr(self, cat_name)
                
                # Update the source_ids and put it back
                cat['source_id'] = cat_source_ids
                setattr(self, cat_name, cat)
                
                del cat, cat_source_ids
                
            del cats
            
            # Plot it
            if plot:
                plt.figure()
                plt.title('{} clusters for {} sources'.format(self.n_sources,len(coords)))
                
                colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, self.n_sources)]
                for k, col in zip(unique_source_ids, colors):
                    
                    class_member_mask = (source_ids == k)
                    xy = coords[class_member_mask & core_samples_mask]
                    
                    marker = 'o'
                    if len(xy)==1:
                        col = [0,0,0,1]
                        marker = '+'
                        
                    plt.plot(xy[:, 0], xy[:, 1], color=tuple(col), marker=marker, markerfacecolor=tuple(col))