def as_dictionary(self):
        """
        Convert this object to a dictionary with formatting appropriate for a PIF.

        :returns: Dictionary with the content of this object formatted for a PIF.
        """
        return {to_camel_case(i): Serializable._convert_to_dictionary(self.__dict__[i])
                for i in self.__dict__ if self.__dict__[i] is not None}