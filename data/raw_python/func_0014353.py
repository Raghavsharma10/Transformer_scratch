def delete(self, space_id):
        """
        Deletes a space by ID.
        """

        try:
            self.space_id = space_id
            return super(SpacesProxy, self).delete(space_id)
        finally:
            self.space_id = None