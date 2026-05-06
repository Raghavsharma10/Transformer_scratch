def is_type(self):
        """
        :return:
        :rtype: bool
        """

        if self.__is_type_result is not None:
            return self.__is_type_result

        self.__is_type_result = self.__is_type()

        return self.__is_type_result