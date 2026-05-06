def find(self, space_id, query=None, **kwargs):
        """
        Gets a space by ID.
        """

        try:
            self.space_id = space_id
            return super(SpacesProxy, self).find(space_id, query=query)
        finally:
            self.space_id = None