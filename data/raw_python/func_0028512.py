def index(self, value):
        """
        Args:
            value: index

        Returns: index of the values

        Raises:
            ValueError: value is not in list
        """

        for i, x in enumerate(self):
            if x == value:
                return i

        raise ValueError("{} is not in list".format(value))