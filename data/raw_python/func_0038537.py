def inventory(self, source_id):
        """
        Look at the inventory for a given source
        
        Parameters
        ----------
        source_id: int
            The id of the source to inspect
        """
        if self.n_sources==0:
            print('Please run group_sources() to create the catalog first.')
        
        else:
            
            if source_id>self.n_sources or source_id<1 or not isinstance(source_id, int):
                print('Please enter an integer between 1 and',self.n_sources)
            
            else:
            
                print('Source:')
                print(at.Table.from_pandas(self.catalog[self.catalog['id']==source_id]).pprint())
                for cat_name in self.catalogs:
                    cat = getattr(self, cat_name)
                    rows = cat[cat['source_id']==source_id]
                    if not rows.empty:
                        print('\n{}:'.format(cat_name))
                        at.Table.from_pandas(rows).pprint()