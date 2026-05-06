def drop_catalog(self, cat_name):
        """
        Remove an imported catalog from the Dataset object
        
        Parameters
        ----------
        cat_name: str
            The name given to the catalog
        """
        # Delete the name and data
        self.catalogs.pop(cat_name)
        delattr(self, cat_name)
        
        # Update history
        print("Deleted {} catalog.".format(cat_name))
        self.history += "\n{}: Deleted {} catalog.".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), cat_name)