def get_relationships(self):
        """Gets the relationship list resulting from a search.

        return: (osid.relationship.RelationshipList) - the relationship
                list
        raise:  IllegalState - list already retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.RelationshipList(self._results, runtime=self._runtime)