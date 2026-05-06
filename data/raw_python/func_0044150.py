def rows(self):
        """Return the list of object on the active page."""

        return map(
            lambda o: self._meta.row_class(self, o),
            self.paginator.page(self._meta.page).object_list )