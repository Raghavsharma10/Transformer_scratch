def str_to_date(self):
        """
        Returns the date attribute as a date object.

        :returns: Date of the status if it exists.
        :rtype: date or NoneType
        """

        if hasattr(self, 'date'):
            return date(*list(map(int, self.date.split('-'))))
        else:
            return None