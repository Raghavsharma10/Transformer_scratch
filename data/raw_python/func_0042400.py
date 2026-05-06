def values(self, name):
        """
        RETURN VALUES FOR THE GIVEN PATH NAME
        :param name:
        :return:
        """
        return list(self.lookup_variables.get(unnest_path(name), Null))