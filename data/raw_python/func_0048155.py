def get_comments(self):
        """Gets the comment list resulting from a search.

        return: (osid.commenting.CommentList) - the comment list
        raise:  IllegalState - list has already been retrieved
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.retrieved:
            raise errors.IllegalState('List has already been retrieved.')
        self.retrieved = True
        return objects.CommentList(self._results, runtime=self._runtime)