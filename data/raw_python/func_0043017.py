def not_left(self, num):
        """
        NOT REQUIRED, EXISTS AS OPPOSITE OF not_right()
        """
        if num == None:
            return FlatList([_get_list(self)[-1]])
        if num <= 0:
            return self

        return FlatList(_get_list(self)[num::])