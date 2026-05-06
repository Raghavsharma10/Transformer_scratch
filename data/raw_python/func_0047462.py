def get_families(self):
        """Gets the family list resulting from a search.

        return: (osid.relationship.FamilyList) - the family list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.FamilyList(self._results, runtime=self._runtime)