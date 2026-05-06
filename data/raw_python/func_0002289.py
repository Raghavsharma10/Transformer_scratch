def move_to_placeholder(self, placeholder, sort_order=None):
        """
        .. versionadded: 1.0.2 Move the entire queryset to a new object.

        Returns a queryset with the newly created objects.
        """
        qs = self.all()  # Get clone
        for item in qs:
            # Change the item directly in the resultset.
            item.move_to_placeholder(placeholder, sort_order=sort_order)
            if sort_order is not None:
                sort_order += 1

        return qs