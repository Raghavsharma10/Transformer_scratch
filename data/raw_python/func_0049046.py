def clear_rating(self):
        """Clears the rating.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_avatar_template
        if (self.get_rating_metadata().is_read_only() or
                self.get_rating_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['ratingId'] = self._rating_default