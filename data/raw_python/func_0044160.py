def sort_url(self):
        """
        Return the URL to sort the linked table by this column. If the
        table is already sorted by this column, the order is reversed.

        Since there is no canonical URL for a table the current URL (via
        the HttpRequest linked to the Table instance) is reused, and any
        unrelated parameters will be included in the output.
        """

        prefix = (self.sort_direction == "asc") and "-" or ""
        return self.table.get_url(order_by=prefix + self.name)