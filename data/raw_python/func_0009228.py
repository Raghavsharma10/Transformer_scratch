def __normalize_grades(self):
        """
        Adjust the grades list.
        If a grade has been set, set All to false
        """

        if 'grades' in self and self['grades']['All'] is True:
            for grade in self['grades']:
                if grade != 'All' and self['grades'][grade] is True:
                    self['grades']['All'] = False
                    break