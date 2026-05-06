def set_end_point_uri(self) -> bool:
        """
        Extracts the route from the accessed URL and sets it to __end_point_uri
        :rtype: bool
        """
        expected_parts = self.__route.split("/")
        actual_parts = self.__uri.split("/")

        i = 0
        for part in expected_parts:
            if part != actual_parts[i]:
                return False
            i = i + 1

        uri_prefix = len(self.__route)
        self.__end_point_uri = self.__uri[uri_prefix:]
        return True