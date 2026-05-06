def sort_direction(self):
        """
        Return the direction in which the linked table is is sorted by
        this column ("asc" or "desc"), or None this column is unsorted.
        """

        if self.table._meta.order_by == self.name:
            return "asc"

        elif self.table._meta.order_by == ("-" + self.name):
            return "desc"

        else:
            return None