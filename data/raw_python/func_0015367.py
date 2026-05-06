def data_x_range(self):
        """Return a 2-tuple giving the minimum and maximum x-axis
        data range.
        """
        try:
            lower = min([min(self._filter_none(s))
                         for type, s in self.annotated_data()
                         if type == 'x'])
            upper = max([max(self._filter_none(s))
                         for type, s in self.annotated_data()
                         if type == 'x'])
            return (lower, upper)
        except ValueError:
            return None