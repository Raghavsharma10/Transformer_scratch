def left(self, num=None):
        """
        NOT REQUIRED, BUT EXISTS AS OPPOSITE OF right()
        """
        if num == None:
            return FlatList([_get_list(self)[0]])
        if num <= 0:
            return Null

        return FlatList(_get_list(self)[:num])