def line_items(self):
        """Apply a datetime filter against the contributors's line item queryset."""
        if self._line_items is None:
            self._line_items = self.contributor.line_items.filter(
                payment_date__range=(self.start, self.end)
            )
        return self._line_items