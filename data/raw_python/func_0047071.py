def get_catalogs(self):
        """Gets the catalog list resulting from the search.

        return: (osid.cataloging.CatalogList) - the catalogs list
        raise:  IllegalState - list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.CatalogList(self._results, runtime=self._runtime)