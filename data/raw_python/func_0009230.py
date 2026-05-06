def __normalize(self):
        """
        Adjusts the values of the filters to be correct.
        For example, if you set grade 'B' to True, then 'All'
        should be set to False
        """

        # Don't normalize if we're already normalizing or intializing
        if self.__normalizing is True or self.__initialized is False:
            return

        self.__normalizing = True
        self.__normalize_grades()
        self.__normalize_progress()
        self.__normalizing = False