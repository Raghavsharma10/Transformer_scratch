def copy_to_placeholder(self, placeholder, sort_order=None):
        """
        .. versionadded: 1.0 Copy the entire queryset to a new object.

        Returns a queryset with the newly created objects.
        """
        qs = self.all()  # Get clone
        for item in qs:
            # Change the item directly in the resultset.
            item.copy_to_placeholder(placeholder, sort_order=sort_order, in_place=True)
            if sort_order is not None:
                sort_order += 1

        return qs