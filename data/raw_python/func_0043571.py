def insert(self, index, p_object):
        """
        Insert an element to a list
        """
        validated_value = self.get_validated_object(p_object)
        if validated_value is not None:
            self.__modified_data__.insert(index, validated_value)