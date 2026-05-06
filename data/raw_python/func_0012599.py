def by_field(self, field):
        """
        Returns an iterator that iterates over unique values of field

        Parameters:
            field : string
                Filters the datamat for every unique value in field and yields
                the filtered datamat.
        Returns:
            datamat : Datamat that is filtered according to one of the unique
                values in 'field'.
        """
        for value in np.unique(self.__dict__[field]):
            yield self.filter(self.__dict__[field] == value)