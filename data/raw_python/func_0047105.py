def add_days(self, *days):
        """Add one or several days to the program.
    
        Parameters
        ----------
        *days
            Unpacked tuple containing
            :py:class:`streprogen.Day` instances.
    
    
        Examples
        -------
        >>> program = Program('My training program')
        >>> day1, day2 = Day(), Day()
        >>> program.add_days(day1, day2)
        """
        for day in list(days):
            self.days.append(day)