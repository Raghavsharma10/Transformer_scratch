def _catalog_check(self, cat_name, append=False):
        """
        Check to see if the name of the ingested catalog is valid
        
        Parameters
        ----------
        cat_name: str
            The name of the catalog in the Catalog object
        append: bool
            Append the catalog rather than replace
        
        Returns
        -------
        bool
            True if good catalog name else False
        """
        good = True
        
        # Make sure the attribute name is good
        if cat_name[0].isdigit():
            print("No names beginning with numbers please!")
            good = False
            
        # Make sure catalog is unique
        if not append and cat_name in self.catalogs:
            print("Catalog {} already ingested. Set 'append=True' to add more records.".format(cat_name))
            good = False
        
        return good