def get_books(self):
        """Gets the book list resulting from a search.

        return: (osid.commenting.BookList) - the book list
        raise:  IllegalState - list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.BookList(self._results, runtime=self._runtime)