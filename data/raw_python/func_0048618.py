def get_items(self):
        """Gets the item list resulting from the search.

        return: (osid.assessment.ItemList) - the item list
        raise:  IllegalState - the item list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.ItemList(self._results, runtime=self._runtime)