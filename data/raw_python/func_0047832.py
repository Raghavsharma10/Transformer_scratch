def get_assets(self):
        """Gets the asset list resulting from a search.

        return: (osid.repository.AssetList) - the asset list
        raise:  IllegalState - the list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.AssetList(self._results, runtime=self._runtime)