def delete_source(self, id):
        """
        Delete a source from the catalog
        
        Parameters
        ----------
        id: int
            The id of the source in the catalog
        """
        # Set the index
        self.catalog.set_index('id')
        
        # Exclude the unwanted source
        self.catalog = self.catalog[self.catalog.id!=id]
        
        # Remove the records from the catalogs
        for cat_name in self.catalogs:
            new_cat = getattr(self, cat_name)[getattr(self, cat_name).source_id!=id]
            print('{} records removed from {} catalog'.format(int(len(getattr(self, cat_name))-len(new_cat)), cat_name))
            setattr(self, cat_name, new_cat)