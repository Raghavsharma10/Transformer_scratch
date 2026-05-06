def append(self, item):
        """
        Appending elements to our list
        """
        validated_value = self.get_validated_object(item)
        if validated_value is not None:
            self.__modified_data__.append(validated_value)