def get(self,key,default=None):
        """Get a value from the dictionary.

        Args:
            key (str): The dictionary key.
            default (any): The default to return if the key is not in the
                dictionary. Defaults to None.

        Returns:
            str or any: The dictionary value or the default if the key is not
                in the dictionary.
        """

        retval = self.__getitem__(key)
        if not retval:
            retval = default

        return retval